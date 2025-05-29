import numpy as np
from evaluate import load
import nltk
nltk.download('punkt')
import ot
import spacy
import ginza
from scipy.optimize import linear_sum_assignment

import torch

from utility.utility_class import UtilityFunction


class OTUtility(UtilityFunction):
    def __init__(self, sentence_sim, ot_alg=None, weight=None, lang='en', gpu=True):
        assert isinstance(sentence_sim, UtilityFunction)
        # assert ot_alg in [None, 'exact', 'sinkhorn']
        # assert weight in [None, 'uniform', 'length']
        self.similarity = sentence_sim
        self.ot_alg = ot_alg
        self.weight = weight
        self.lang = lang
        self.gpu = gpu

        # if self.lang == 'ja':
        #     self.nlp = spacy.load('ja_ginza')
        #     self.nlp.disable_pipes(['tok2vec', 'parser', 'ner', 'morphologizer', 'compound_splitter', 'bunsetu_recognizer'])
        #     self.nlp.add_pipe('sentencizer')
        
    def compute_similarity(self, hyp, ref, src=None):
        n_texts = len(hyp)
        
        hyp_sentences = [self.text2sentences(h) for h in hyp]
        ref_sentences = [self.text2sentences(r) for r in ref]
        # TODO: We don't consider source sentences for now for simplicity.
        #       This mean we cannot use reference-based evaluator like COMET.
        # src_sents = self.text2sentences(src)
        
        scores = []
        for i in range(n_texts):
            if hyp[i] == ref[i]:
                scores.append(1.0)
                continue

            hyp_sents = hyp_sentences[i]
            ref_sents = ref_sentences[i]
            
            cost_matrix = np.zeros((len(hyp_sents), len(ref_sents)))
            for j, hyp_sent in enumerate(hyp_sents):
                sims = self.similarity.compute_similarity([hyp_sent] * len(ref_sents), ref_sents, None)
                cost_matrix[j] = np.subtract(np.ones(len(ref_sents), dtype=np.float32), np.array(sims))

            # TODO: The similarity function should be 0 to 1.
            assert np.all(cost_matrix >= -0.5)
            assert np.all(cost_matrix <= 1.5)
            
            hyp_weights, ref_weights = self.get_weights(hyp_sents, ref_sents)
            # Computing the Optimal Trasport is the bottleneck of the process.
            cost = self.compute_ot(hyp_weights, ref_weights, cost_matrix)
            score = 1.0 - cost
            scores.append(score)
        return scores

    def text2sentences(self, text):
        # This is applicable for WMT datasets.
        sentences = text.split(' </s> ')
        # print('sentences=', sentences)
        return sentences
        # if (self.lang == 'en'):
        #     return nltk.sent_tokenize(text)
        # elif self.lang == 'de':
        #     return nltk.sent_tokenize(text, language='german')
        # elif self.lang == 'es':
        #     return nltk.sent_tokenize(text, language='spanish')
        # elif self.lang == 'ja':
        #     doc = self.nlp(text)
        #     return [sent.text for sent in doc.sents]
        # else:
        #     raise ValueError(f"Invalid language: {self.lang}")

    def get_weights(self, hyp_sents, ref_sents):
        if (self.weight is None) or (self.weight == 'uniform'):
            hyp_weights = np.ones(len(hyp_sents)) / len(hyp_sents)
            ref_weights = np.ones(len(ref_sents)) / len(ref_sents)
        elif self.weight == 'length':
            hyp_weights = np.array([len(s) for s in hyp_sents])
            hyp_weights = hyp_weights / np.sum(hyp_weights)
            ref_weights = np.array([len(s) for s in ref_sents])
            ref_weights = ref_weights / np.sum(ref_weights)
        else:
            raise ValueError(f"Invalid weight type: {self.weight}")
        return hyp_weights, ref_weights
    
    def compute_ot_cost(self, hyp_weights, ref_weights, cost_matrix):
        if (self.ot_alg is None) or (self.ot_alg == 'sinkhorn'):
            cost = ot.sinkhorn2(hyp_weights, ref_weights, cost_matrix, 0.1)
        elif self.ot_alg == 'exact':
            cost = ot.emd2(hyp_weights, ref_weights, cost_matrix)
        elif self.ot_alg == 'greedy':
            cost = ot.bregman.greenkhorn(hyp_weights, ref_weights, cost_matrix, 0.1)
        elif self.ot_alg == 'assign':
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # print(row_ind, col_ind)
            cost = cost_matrix[row_ind, col_ind].mean()
        else:
            raise ValueError(f"Invalid OT algorithm: {self.ot_alg}")
        return cost

    def compute_ot(self, hyp_weights, ref_weights, cost_matrix):
        # assert isinstance(hyp_weights, np.ndarray)
        # assert isinstance(ref_weights, np.ndarray)
        # assert isinstance(cost_matrix, np.ndarray)
        if self.gpu:
            hyp_weights_torch = torch.tensor(hyp_weights, dtype=torch.float32, device='cuda')
            ref_weights_torch = torch.tensor(ref_weights, dtype=torch.float32, device='cuda')
            cost_matrix_torch = torch.tensor(cost_matrix, dtype=torch.float32, device='cuda')
            cost_torch = self.compute_ot_cost(hyp_weights_torch, ref_weights_torch, cost_matrix_torch)
            cost = cost_torch.cpu().detach().item()
        else:
            cost = self.compute_ot_cost(hyp_weights, ref_weights, cost_matrix)
        return cost

    def set_lang(self, lang: str):
        self.lang = lang
        if self.lang == 'ja':
            self.nlp = spacy.load('ja_ginza')

    def set_gpu(self, gpu: bool):
        self.gpu = gpu