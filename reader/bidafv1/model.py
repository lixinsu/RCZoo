#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA Document Reader model"""

import math
import random

import ipdb
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy

from torch.autograd import Variable
from .config import override_model_args
from .rnn_reader import RnnDocReader

logger = logging.getLogger(__name__)



class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, char_dict, feature_dict,
                 state_dict=None, normalize=True):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.char_dict = char_dict
        self.args.vocab_size = len(word_dict)
        self.args.char_vocab_size = len(char_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            self.network.load_state_dict(state_dict)

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert (len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True))
                      for e in ex[:6]]
            target_s = Variable(ex[6].cuda(async=True))
            target_e = Variable(ex[7].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:5]]
            target_s = Variable(ex[6])
            target_e = Variable(ex[7])

        # Run forward
        score_s, score_e = self.network(*inputs)

        if self.args.smooth == 'gauss':
            # label smoothing
            class GussianNoise(object):
                def __init__(self, mu, sigma):
                    self.mu = mu
                    self.sigma = sigma

                def pdf(self, x):
                    return 1.0 / math.sqrt(
                        2 * math.pi * self.sigma * self.sigma) * math.exp(
                        -(x - self.mu) * (
                            x - self.mu) / 2 / self.sigma / self.sigma)

                def get_prob(self, x):
                    return self.pdf(x)

                def get_probs(self, n):
                    return np.array([self.get_prob(x) for x in range(n)])

            doc_lenths = (ex[2].size(1) - ex[2].sum(dim=1)).tolist()
            answer_lengths = (target_e + 1 - target_s).tolist()
            start_mu = target_s.tolist()
            end_mu = target_e.tolist()
            start_proba = []
            end_proba = []
            paded_doc_length = ex[2].size(1)
            for s, e, sigma, doc_len in zip(start_mu, end_mu, answer_lengths,
                                            doc_lenths):
                start_proba.append(
                    GussianNoise(s, sigma * self.args.smooth_scale).get_probs(
                        paded_doc_length))
                end_proba.append(
                    GussianNoise(e, sigma * self.args.smooth_scale).get_probs(
                        paded_doc_length))
            start_proba = torch.Tensor(start_proba).cuda()
            end_proba = torch.Tensor(end_proba).cuda()

            if self.args.add_main:
                # Add main
                main_s = torch.zeros(score_e.size()).cuda()
                main_e = torch.zeros(score_e.size()).cuda()
                main_s.scatter_(1, target_s.unsqueeze(1), 1)
                main_e.scatter_(1, target_e.unsqueeze(1), 1)
                start_proba += main_s
                end_proba += main_e

            # previous normalization
            start_proba.masked_fill_(ex[2].cuda(), 0)
            end_proba.masked_fill_(ex[2].cuda(), 0)
            start_proba = start_proba / start_proba.sum(dim=1).unsqueeze(1)
            end_proba = end_proba / end_proba.sum(dim=1).unsqueeze(1)

            loss = F.kl_div(score_s, start_proba,
                            reduction='batchmean') + F.kl_div(score_e,
                                                              end_proba,
                                                              reduction='batchmean')
            if self.args.multiloss:
                loss = loss * self.args.newloss_scale + F.nll_loss(score_s,
                                                                   target_s) + F.nll_loss(
                    score_e,
                    target_e)

        elif self.args.smooth == 'smooth':
            alpha = self.args.normal_alpha
            main_s = torch.zeros(score_e.size()).cuda()
            main_e = torch.zeros(score_e.size()).cuda()
            main_s.scatter_(1, target_s.unsqueeze(1), 1)
            main_e.scatter_(1, target_e.unsqueeze(1), 1)
            start = torch.ones(score_e.size())
            start.masked_fill_(ex[2], 0)
            start = start / start.sum(dim=-1, keepdim=True)
            start = start.cuda()
            start_gt = main_s * (1 - alpha) + alpha * start
            end_gt = main_e * (1 - alpha) + alpha * start
            loss = torch.sum(- start_gt * score_s, -1) + \
                   torch.sum(- end_gt * score_e, -1)
            loss = loss.mean()

        elif self.args.smooth == 'xxxx':
            def f1(s, e, s_, e_):
                gt = set(range(s, e + 1))
                pr = set(range(s_, e_ + 1))
                common = gt & pr
                if len(common) == 0:
                    return 0
                p = len(common) / len(pr)
                r = len(common) / len(gt)
                return 2 * p * r / (p + r)

            start_idx = torch.multinomial(torch.exp(score_s), 1)
            end_idx = torch.multinomial(torch.exp(score_e), 1)
            start_idx = start_idx.flatten()
            end_idx = end_idx.flatten()

            cpu_start_idx = start_idx.tolist()
            cpu_end_idx = end_idx.tolist()

            greedy_start_idx = torch.argmax(score_s, dim=1).tolist()
            greedy_end_idx = torch.argmax(score_e, dim=1).tolist()

            gt_start = target_s.tolist()
            gt_end = target_e.tolist()

            base_rewards = []
            for s, e, s_, e_ in zip(gt_start, gt_end, greedy_start_idx,
                                    greedy_end_idx):
                base_rewards.append(f1(s, e, s_, e_))
            base_rewards = torch.Tensor(base_rewards).cuda()

            rewards = []
            for s, e, s_, e_ in zip(gt_start, gt_end, cpu_start_idx,
                                    cpu_end_idx):
                rewards.append(f1(s, e, s_, e_))
            rewards = torch.Tensor(rewards).cuda()

            mle_loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e,
                                                                  target_e)
            augment_loss = F.nll_loss(score_s, start_idx, reduction='none') + \
                           F.nll_loss(score_e, end_idx, reduction='none')

            augment_loss *= (rewards - base_rewards)

            loss = (1 - self.args.newloss_scale) * mle_loss + \
                   self.args.newloss_scale * augment_loss.mean()


        elif self.args.smooth == 'dcrl' or  self.args.smooth == 'scst':
            def mask_to_start(score, start, score_mask_value=-1e30):
                score_mask = (torch.ones_like(start).cuda() - torch.cumsum(start, dim=-1)).float()
                return score + score_mask * score_mask_value

            def get_f1(y_pred, y_true):
                y_true = y_true.float()
                y_pred = y_pred.float()
                y_union = torch.clamp(y_pred + y_true, 0, 1)  # [bs, seq]
                y_diff = torch.abs(y_pred - y_true)  # [bs, seq]
                num_same = (torch.sum(y_union, dim=-1) - torch.sum(y_diff, dim=-1)).float()  # [bs,]
                y_precision = num_same / (torch.sum(y_pred, dim=-1).float() + 1e-7)  # [bs,]
                y_recall = num_same / (torch.sum(y_true, dim=-1).float() + 1e-7)  # [bs,]
                y_f1 = (2.0 * y_precision * y_recall) / ((y_precision + y_recall).float() + 1e-7)  # [bs,]
                return torch.clamp(y_f1, 0, 1)

            def one_hot(tensor):
                rv = torch.LongTensor(score_s.size()).cuda()
                rv.zero_()
                rv.scatter_(1, tensor.unsqueeze(1), 1)
                return rv
            #start_one_hot = tf.one_hot(start_positions, depth=seq_length)
            #start_one_hot = torch.LongTensor(*score_s.size())
            #start_one_hot.zero_()
            #start_one_hot.scatter_(1, target_s, 1)
            import ipdb
            ipdb.set_trace()
            start_one_hot = one_hot(target_s)

            #end_one_hot = tf.one_hot(end_positions, depth=seq_length)
            #end_one_hot = torch.LongTensor(*score_s.size())
            #end_one_hot.zero_()
            #end_one_hot.scatter_(1, target_e, 1)
            end_one_hot = one_hot(target_e)

            #start_cumsum = tf.cumsum(start_one_hot, axis=-1)
            #end_cumsum = tf.cumsum(end_one_hot, axis=-1)
            start_cumsum = torch.cumsum(start_one_hot, dim=1)
            end_cumsum = torch.cumsum(end_one_hot, dim=1)

            ground_truth = start_cumsum - end_cumsum + end_one_hot

            #greedy_start = one_hot(tf.argmax(score_s, axis=-1))
            greedy_start_ind = torch.argmax(score_s, dim=-1)
            greedy_start = one_hot(greedy_start_ind)
            masked_end_logits = mask_to_start(score_e, greedy_start)
            #greedy_end = one_hot(tf.argmax(masked_end_logits, axis=-1))
            greedy_end_ind = torch.argmax(masked_end_logits, dim=-1)
            greedy_end = one_hot(greedy_end_ind)
            #greedy_start_cumsum = tf.cumsum(greedy_start, axis=-1)
            #greedy_end_cumsum = tf.cumsum(greedy_end, axis=-1)

            greedy_start_cumsum = torch.cumsum(greedy_start, dim=-1)
            greedy_end_cumsum = torch.cumsum(greedy_end, dim=-1)

            greedy_prediction = greedy_start_cumsum - greedy_end_cumsum + greedy_end
            greedy_f1 = get_f1(greedy_prediction, ground_truth)

            sampled_start_ind = torch.multinomial(torch.softmax(score_s, dim=-1), 1).squeeze()
            sampled_start = one_hot(sampled_start_ind)
            masked_end_logits = mask_to_start(score_e, sampled_start)
            sampled_end_ind = torch.multinomial(torch.softmax(masked_end_logits, dim=-1), 1).squeeze()
            sampled_end = one_hot(sampled_end_ind)
            sampled_start_cumsum = torch.cumsum(sampled_start, dim=-1)
            sampled_end_cumsum = torch.cumsum(sampled_end, dim=-1)
            sampled_prediction = sampled_start_cumsum - sampled_end_cumsum + sampled_end
            sampled_f1 = get_f1(sampled_prediction, ground_truth)
            reward = sampled_f1 - greedy_f1

            sampled_start_loss = F.cross_entropy(score_s, sampled_start_ind)
            sampled_end_loss = F.cross_entropy(score_e, sampled_end_ind)

            greedy_start_loss = F.cross_entropy(score_s, greedy_start_ind)
            greedy_end_loss = F.cross_entropy(score_e, greedy_end_ind)

            mle_loss = F.cross_entropy(score_s, target_s) + F.cross_entropy(score_e, target_e)
            if  self.args.smooth == 'dcrl':
                reward = torch.clamp(reward, 0., 1e7)
                reward_greedy = torch.clamp(greedy_f1 - sampled_f1, 0., 1e7)
                rl_loss = (reward * (sampled_start_loss + sampled_end_loss) + reward_greedy * (
                                    greedy_start_loss + greedy_end_loss)).mean()
            else:
                rl_loss = (reward * (sampled_start_loss + sampled_end_loss)).mean()
            loss = (1 - self.args.alpha) * mle_loss + self.args.alpha * rl_loss

        elif self.args.smooth == 'reward':
            def f1(s, e, s_, e_):
                gt = set(range(s, e + 1))
                pr = set(range(s_, e_ + 1))
                common = gt & pr
                if len(common) == 0:
                    return 0
                p = len(common) / len(pr)
                r = len(common) / len(gt)
                return 2 * p * r / (p + r)

            def calculate_reward(s, e, n, pad_n, val=-2000):
                start = [val] * pad_n
                end = [val] * pad_n
                for i in range(0, e + 1):
                    start[i] = f1(s, e, i, e)
                for i in range(s, n):
                    end[i] = f1(s, e, s, i)
                return start, end

            def softmax(li, T=0.5):
                exp_li = [math.exp(x / T) for x in li]
                nomi = sum(exp_li)
                return [x / nomi for x in exp_li]

            def make_proba(li):
                nomi = sum(li)
                return [x / nomi for x in li]

            start_mu = target_s.tolist()
            end_mu = target_e.tolist()
            doc_lengths = (ex[2].size(1) - ex[2].sum(dim=1)).tolist()
            start_gt = []
            end_gt = []
            for s, e, n in zip(start_mu, end_mu, doc_lengths):
                start_, end_ = calculate_reward(s, e, n, ex[2].size(1))
                start_gt.append(softmax(start_, self.args.temperature))
                end_gt.append(softmax(end_, self.args.temperature))
            start_gt = torch.Tensor(start_gt).cuda()
            end_gt = torch.Tensor(end_gt).cuda()

            def cross_entropy(log_proba, gt):
                return torch.sum( - gt * log_proba, dim=1 ).mean()

            rls_loss = cross_entropy(score_s, start_gt) + cross_entropy(score_e, end_gt)
            mle_loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            loss = rls_loss * self.args.alpha + (1-self.args.alpha) * mle_loss

        elif self.args.smooth == 'ce':
            # Compute loss and accuracies
            loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e,
                                                              target_e)

        else:
            raise ValueError("Undefine loss")
        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.network.named_parameters():
        #    if param.requires_grad:
        #        print("-"*40,name,"-"*40)
        #        print(torch.sum(param.grad))
        #        print(torch.sum(torch.abs(param.grad)))
        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.item(), ex[0].size(0)

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding

            # Embeddings to fix are the last indices
            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0:
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """Forward a batch of examples only to get predictions.

        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True))
                      for e in ex[:6]]
            gt_s = [x.item() for x in ex[6]]
            gt_e = [x.item() for x in ex[7]]
            target_s = torch.LongTensor(gt_s).cuda()
            target_e = torch.LongTensor(gt_e).cuda()
        else:
            inputs = [e if e is None else Variable(e)
                      for e in ex[:6]]
            gt_s = [x[0] for x in ex[6]]
            gt_e = [x[0] for x in ex[7]]
            target_s = torch.LongTensor(gt_s)
            target_e = torch.LongTensor(gt_e)

        # Run forward
        score_s, score_e = self.network(*inputs)

        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)

        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_s, score_e, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args), loss.item()

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []
        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                # try getting from globals? (multiprocessing in pipeline mode)
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')

            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    # Match! Record its score.
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)

            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'char_dict': self.char_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'word_dict': self.word_dict,
            'char_dict': self.char_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        char_dict = saved_params['char_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return DocReader(args, word_dict, char_dict, feature_dict, state_dict,
                         normalize)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        char_dict = saved_params['char_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = DocReader(args, word_dict, char_dict, feature_dict, state_dict,
                          normalize)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
