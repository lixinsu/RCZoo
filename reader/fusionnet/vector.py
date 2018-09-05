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
    word_dict = model.word_dict
    char_dict = model.char_dict
    pos_dict = model.pos_dict
    ner_dict = model.ner_dict

    # Index words, pos, ner
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    d_pos = torch.LongTensor([pos_dict[ipos] for ipos in ex['pos']])
    d_ner = torch.LongTensor([ner_dict[iner] for iner in ex['ner']])
    q_pos = torch.LongTensor([pos_dict[ipos] for ipos in ex['q_pos']])
    q_ner = torch.LongTensor([ner_dict[iner] for iner in ex['q_ner']])

    # Index chars and padding to word_len
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

    # Index mannual feature
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

    if single_answer:
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return ex['document'], ex['question'], document, document_char, d_pos, d_ner, question, question_char, q_pos, q_ner, features, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    ids = [ex[-1] for ex in batch]

    d = [ex[2] for ex in batch]
    dc = [ex[3] for ex in batch]
    dp = [ex[4] for ex in batch]
    de = [ex[5] for ex in batch]
    df = [ex[10] for ex in batch]
    word_len = dc[0].size(1)

    df = [ex[10] for ex in batch]

    q = [ex[6] for ex in batch]
    qc = [ex[7] for ex in batch]
    qp = [ex[8] for ex in batch]
    qe = [ex[9] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_p = torch.LongTensor(len(docs), max_length).zero_()
    x1_e = torch.LongTensor(len(docs), max_length).zero_()
    x1_c = torch.LongTensor(len(docs), max_length, word_len).zero_()

    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))

    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)

    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_p[i, :d.size(0)].copy_(dp)
        x1_e[i, :d.size(0))].copy_(de)
        x1_c[i, :d.size(0)].copy_(docs_char[i])
        x1_f[i, :d.size(0)].copy_(df[i])
        x1_mask[i, :d.size(0)].fill_(0)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_p = torch.LongTensor(len(questions), max_length).zero_()
    x2_e = torch.LongTensor(len(questions), max_length).zero_()
    x2_c = torch.LongTensor(len(questions), max_length, word_len).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_p[i, :q.size(0)].copy_(qp)
        x2_e[i, :q.size(0)].copy_(qe)
        x2_c[i, :q.size(0)].copy_(questions_char[i])
        x2_mask[i, :q.size(0)].fill_(0)

    if torch.is_tensor(batch[0][4]):
        y_s = torch.cat([ex[4] for ex in batch])
        y_e = torch.cat([ex[5] for ex in batch])
    else:
        y_s = [ex[4] for ex in batch]
        y_e = [ex[5] for ex in batch]

    return x1, x1_p, x1_e, x1_c, x1_mask, x2, x2_p, x2_e, x2_c, x2_mask, x1_f ,y_s, y_e, ids
