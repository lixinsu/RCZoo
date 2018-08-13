#!/usr/bin/env python
# coding: utf-8
import os
import re
import copy
from pyltp import Segmentor, Postagger, NamedEntityRecognizer
from .tokenizer import Tokens, Tokenizer


LTP_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ltp_data')
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')

class LtpTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:

        """
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(cws_model_path)
        self.postagger = Postagger() # 初始化实例
        self.postagger.load(pos_model_path)  # 加载模型
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(ner_model_path)  # 加载模型
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
    def tokenize(self, text):
        clean_text = text.replace(' ', ',') # for ltp process ,ltp 不把空格当成词的边界
        tokens = list(self.segmentor.segment(clean_text)) # 分词
        postags = list(self.postagger.postag(tokens))     # 词性标注
        netags = list(self.recognizer.recognize(tokens, postags))  # 命名实体识别
        idxs = []
        j = 0
        i = 0
        #print(text)
        #print(tokens)
        #print(postags)
        #print(netags)
        while i <  len(tokens):
            #print(clean_text[j:j+len(tokens[i])], tokens[i])
            if clean_text[j:j+len(tokens[i])] == tokens[i]:
                idxs.append(j)
                j += len(tokens[i])
                i += 1
            else:
                j += 1
            #print(i,j)
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
                postags[i],
                tokens[i],
                netags[i],
                ))
        return Tokens(data, self.annotators)



