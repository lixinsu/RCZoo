#!/usr/bin/env python
# coding: utf-8

import jieba
import copy
import jieba.posseg as pseg
from .tokenizer import Tokens, Tokenizer

#jieba.load_userdict('/home/sulixin/relate/DrQA_Sougou/data/user_dict_ans.txt')
class JiebaTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))

    def tokenize(self, text):
        # discard newline
        clean_text = text.replace('\n', ' ')
        tokens, poss = [], []
        for tok,pos in pseg.cut(clean_text):
            tokens.append(tok)
            poss.append(pos)
        idxs = []
        j = 0
        i = 0
        while i <  len(tokens):
            if clean_text[j:j+len(tokens[i])] == tokens[i]:
                idxs.append(j)
                j += len(tokens[i])
                i += 1
            else:
                j += 1

        #print(tokens)
        #print(idxs)
        data = []
        for i in range(len(tokens)):
            start_ws = idxs[i]
            if i + 1 < len(tokens):
                end_ws = idxs[i+1]
            else:
                end_ws = idxs[i] + len(tokens[i])
            data.append((
                    tokens[i],
                    text[start_ws:end_ws],
                    (idxs[i], idxs[i] + len(tokens[i])),
                    poss[i],
                    tokens[i],
                    'fake',
                ))
        return Tokens(data, self.annotators)
