import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import json
import os, os.path
import re
import nltk
import ray

import flair
from flair.data import Sentence, Corpus, Token
from flair.datasets import ColumnCorpus
from flair.datasets.sequence_labeling import ColumnDataset
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

@ray.remote
def normalize_empty_lines(file, repl):
    return re.sub(r'^[\s\t]*?\n', repl, file, flags=re.MULTILINE)

@ray.remote(num_gpus=1)
def contains_keywords(line, tagger):
    line_with_label = Sentence(line, nltk.wordpunct_tokenize)
    tagger.predict(line_with_label)
    for j in range(len(line_with_label)):
        found = False
        if line_with_label[j].labels[0].value not in ['other', 'comment']:
            found = True
            break
    return found

# @ray.remote(num_return_vals=3)
def extract_from_file(tagger, files, labels, file_id, keep):
    extracted_lines = []
    extracted_labels = []
    extract_from = []
    
    file = files[file_id]
    if labels:
        label = labels[file_id]
        label = list(label)
    
    count = 0
    i = j = 0
    lines = nltk.line_tokenize(file)

    start = None
    count = 0
    
    found_list = [contains_keywords.remote(line, tagger) for line in lines]
    found_list = ray.get(found_list)
    while i < len(lines):
        print(i)
        line = lines[i]
        if found_list[i]:
            count = 0
            if not start:
                start = i
        else:
            count += 1

        if start is not None:
            if count == keep or i == len(lines) - 1:
                start = max(start-keep, 0)
                end = min(i, len(lines)-1)
#                         print(end)
                new_extracted = '\n'.join(lines[start: end+1])
                extracted_lines.append(new_extracted)
                if labels:
                    extracted_labels.append(label[start: end+1])
                extract_from.append((file_id, start))
                start = None
                count = 0
        i += 1   
    return extracted_lines, extracted_labels, extract_from

def extract_entities(tagger, files, labels=None, keep=2):
    
    extracted_lines_all = []
    extracted_labels_all = []
    extract_from_all = []
    tagger.eval()
    for file_id in range(len(files)):
        print(file_id)
        print(torch.cuda.is_available())
        extracted_lines_single, extracted_labels_single, extract_from_single = \
            extract_from_file(tagger, files, labels, file_id, keep)
        extracted_lines_all.extend(extracted_lines_single)
        if labels:
            extracted_labels_all.extend(extracted_labels_single)
        extract_from_all.extend(extract_from_single)
    return extracted_lines_all, extracted_labels_all, extract_from_all

@ray.remote
def to_sentence(file):
    return Sentence(file, use_tokenizer=nltk.line_tokenize)

@ray.remote
def remote_set_label(sent, label):
    sent.set_label('label', label)

def file_preprocessing(tagger, file_content, line_labels=None):
    file_content = [normalize_empty_lines.remote(file, r'*\n') for file in file_content]
    file_content = ray.get(file_content)
    extracted_lines, extracted_labels, extract_from = \
        extract_entities(tagger, file_content, line_labels)

    file_to_sentences = [to_sentence.remote(file) for file in extracted_lines]
    file_to_sentences = ray.get(file_to_sentences)
    
    if line_labels:
        set_label_agent = [[remote_set_label.remote(sent, label) for sent, label in zip(file, labels)] 
         for file, labels in zip(file_to_sentences, extracted_labels)]
        ray.get(set_label_agent)
    return file_to_sentences, extract_from

bert_embedding = TransformerWordEmbeddings('bert-base-cased', fine_tune=False, allow_long_sentences=True)
# init Flair forward and backwards embeddings
flair_embedding_forward = FlairEmbeddings('mix-forward', fine_tune=False, )
flair_embedding_backward = FlairEmbeddings('mix-backward', fine_tune=False, )

embeddings = StackedEmbeddings([
                                bert_embedding,
                                flair_embedding_forward,
                                flair_embedding_backward,
                               ])
line = 'This is an example . \n The second line .'
sent = Sentence(line)
for token in sent.tokens:
    token.set_label('label', 'other')
sentences = [sent]
lines = [line, line]
corpus = Corpus(train=sentences, dev=[], test=[])
tag_type = 'label'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type=tag_type,
#                                         dropout=0,
                        word_dropout=0.05,
                        locked_dropout=0.1,
                        weight_dropout=0.1,
                       )


ray.shutdown()
ray.init(num_cpus=32, num_gpus=1)

unlabeled_file_to_sentences, unlabeled_extract_from = file_preprocessing(tagger, lines)

