#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    args.word_len = 15
    word_dict = model.word_dict
    char_dict = model.char_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    dc = [[char_dict[c] for c in w] for w in ex['document']]
    for i in range(len(dc)):
        if len(dc[i]) < args.word_len:
            dc[i] = dc[i] + [0] * (args.word_len - len(dc[i]))
        dc[i] = dc[i][:args.word_len]
    document_char = torch.LongTensor(dc)

    qc = [[char_dict[c] for c in w] for w in ex['question']]
    for i in range(len(qc)):
        if len(qc[i]) < args.word_len:
            qc[i] = qc[i] + [0] * (args.word_len - len(qc[i]))
        qc[i] = qc[i][:args.word_len]
    question_char = torch.LongTensor(qc)

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    start = torch.LongTensor(1).fill_(ex['answers'][0][0])
    end = torch.LongTensor(1).fill_(ex['answers'][0][1])

    return document, document_char, question, question_char, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    docs_char = [ex[1] for ex in batch]
    word_len = docs_char[0].size(1)
    questions = [ex[2] for ex in batch]
    questions_char = [ex[3] for ex in batch]
    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    #max_length = 400
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_c = torch.LongTensor(len(docs), max_length, word_len).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_c[i, :d.size(0), :].copy_(docs_char[i])
        x1_mask[i, :d.size(0)].fill_(0)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    #max_length = 20
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_c = torch.LongTensor(len(questions), max_length, word_len).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_c[i, :q.size(0), :].copy_(questions_char[i])
        x2_mask[i, :q.size(0)].fill_(0)

    y_s = torch.cat([ex[4] for ex in batch])
    y_e = torch.cat([ex[5] for ex in batch])
    assert torch.sum(y_s<0) == 0
    assert torch.sum(y_e<0) == 0
    return x1, x1_c, x1_mask, x2, x2_c, x2_mask, y_s, y_e, ids
