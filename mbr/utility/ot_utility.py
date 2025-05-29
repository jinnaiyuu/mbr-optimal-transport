from utility.utility_class import UtilityFunction
import torch
from scipy.optimize import linear_sum_assignment
import ginza
import spacy
import ot
from typing import List, Optional, Any, Union, Tuple

import numpy as np
from evaluate import load
import nltk

nltk.download("punkt")


class OTUtility(UtilityFunction):
    """
    Utility function based on Optimal Transport for comparing text similarity.

    This class implements a text similarity measure using Optimal Transport (OT)
    to align sentences between hypothesis and reference texts. It allows for
    different OT algorithms, weighting schemes, and language-specific processing.
    """

    def __init__(
        self,
        sentence_sim: UtilityFunction,
        ot_alg: Optional[str] = None,
        weight: Optional[str] = None,
        lang: str = "en",
        gpu: bool = True,
    ) -> None:
        """
        Initialize the OT utility function.

        Args:
            sentence_sim: A utility function for computing sentence-level similarity.
            ot_alg: The OT algorithm to use. Options include 'sinkhorn', 'exact', 'greedy', 'assign'.
                    If None, defaults to 'sinkhorn'.
            weight: The weighting scheme for sentences. Options include 'uniform', 'length'.
                    If None, defaults to 'uniform'.
            lang: The language of the texts. Supported languages: 'en', 'de', 'ja'.
            gpu: Whether to use GPU acceleration for OT computation.
        """
        assert isinstance(sentence_sim, UtilityFunction)
        # assert ot_alg in [None, 'exact', 'sinkhorn']
        # assert weight in [None, 'uniform', 'length']
        self.similarity = sentence_sim
        self.ot_alg = ot_alg
        self.weight = weight
        self.lang = lang
        self.gpu = gpu

        if self.lang == "ja":
            self.nlp = spacy.load("ja_ginza")
            self.nlp.disable_pipes(
                [
                    "tok2vec",
                    "parser",
                    "ner",
                    "morphologizer",
                    "compound_splitter",
                    "bunsetu_recognizer",
                ]
            )
            self.nlp.add_pipe("sentencizer")

    def compute_similarity(
        self, hyp: List[str], ref: List[str], src: Optional[Any] = None
    ) -> List[float]:
        """
        Compute similarity scores between hypotheses and references using Optimal Transport.

        Args:
            hyp: List of hypothesis strings.
            ref: List of reference strings.
            src: Optional source input (not used in current implementation).

        Returns:
            List of similarity scores between each hypothesis-reference pair.

        Note:
            This method breaks down texts into sentences, computes a cost matrix
            based on sentence-level similarities, and then solves an optimal transport
            problem to find the best alignment between sentences.
        """
        n_texts = len(hyp)

        # TODO: We don't consider source sentences for now for simplicity.
        #       This mean we cannot use reference-based evaluator like COMET.
        #       One can extend this function to use source sentences by computing the optimal transport
        #       between source sentences and target sentences.
        hyp_sentences = [self.text2sentences(h) for h in hyp]
        ref_sentences = [self.text2sentences(r) for r in ref]

        scores = []
        for i in range(n_texts):
            if hyp[i] == ref[i]:
                scores.append(1.0)
                continue

            hyp_sents = hyp_sentences[i]
            ref_sents = ref_sentences[i]

            # This is a temporary fix for the case where the number of sentences is too different.
            # The problem is often because Japanese sentence segmentation
            # library does not properly divide texts with mixed languages.
            # So we ignore cases where the number of sentences are more than twice different and
            # set their similarity to 0.
            hyp_sents, ref_sents = self.n_sentences_threshold(hyp_sents, ref_sents)

            cost_matrix = np.zeros((len(hyp_sents), len(ref_sents)))
            for j, hyp_sent in enumerate(hyp_sents):
                # NOTE: Cache the embeddings for faster computation.
                #       However, that requires much more VRAM memory which is not available in many setups.
                sims = self.similarity.compute_similarity(
                    [hyp_sent] * len(ref_sents), ref_sents, None
                )
                print(f"hyp: {hyp_sent}, ref: {ref_sents}, sims: {sims}")

                # NOTE: For efficiency, keep the cost matrix at GPU.
                cost_matrix[j] = np.subtract(
                    np.ones(len(ref_sents), dtype=np.float32), np.array(sims)
                )

            # The similarity function should be 0 to 1. 1 means identical, 0 means completely different.
            assert np.all(cost_matrix >= -0.5)
            assert np.all(cost_matrix <= 1.5)

            hyp_weights, ref_weights = self.get_weights(hyp_sents, ref_sents)

            # NOTE: For efficiency, keep the cost matrix at GPU.
            #       It takes time to transfer the cost matrix from GPU to CPU.
            #       However, that requires more VRAM memory.
            cost = self.compute_ot(hyp_weights, ref_weights, cost_matrix)
            score = 1.0 - cost
            scores.append(score)
        return scores

    def n_sentences_threshold(
        self,
        hyp: List[str],
        ref: List[str],
        threshold: float = 2.0,
        max_threshold: int = 32,
    ) -> Tuple[List[str], List[str]]:
        """
        Adjust the number of sentences if they are too different.

        Args:
            hyp: List of hypothesis sentences.
            ref: List of reference sentences.
            threshold: Maximum allowed ratio between the number of sentences.
            max_threshold: Maximum absolute threshold (not used in current implementation).

        Returns:
            Tuple containing adjusted hypothesis and reference sentence lists.

        Note:
            For document pairs with very different numbers of sentences,
            their similarity may not be meaningful anyway.
        """
        l_hyp = len(hyp)
        l_ref = len(ref)
        if l_hyp > threshold * l_ref:
            hyp = hyp[: int(threshold * l_ref)]
        elif l_ref > threshold * l_hyp:
            ref = ref[: int(threshold * l_hyp)]
        return hyp, ref

    def text2sentences(self, text: str) -> List[str]:
        """
        Split text into sentences based on the language.

        Args:
            text: The text to split into sentences.

        Returns:
            List of sentences.

        Raises:
            ValueError: If the language is not supported.
        """
        if self.lang == "en":
            return nltk.sent_tokenize(text)
        elif self.lang == "de":
            return nltk.sent_tokenize(text, language="german")
        elif self.lang == "ja":
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            raise ValueError(f"Invalid language: {self.lang}")

    def get_weights(
        self, hyp_sents: List[str], ref_sents: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weights for sentences based on the specified weighting scheme.

        Args:
            hyp_sents: List of hypothesis sentences.
            ref_sents: List of reference sentences.

        Returns:
            Tuple containing weights for hypothesis and reference sentences.

        Raises:
            ValueError: If the weight type is invalid.
        """
        if (self.weight is None) or (self.weight == "uniform"):
            hyp_weights = np.ones(len(hyp_sents)) / len(hyp_sents)
            ref_weights = np.ones(len(ref_sents)) / len(ref_sents)
        elif self.weight == "length":
            hyp_weights = np.array([len(s) for s in hyp_sents])
            hyp_weights = hyp_weights / np.sum(hyp_weights)
            ref_weights = np.array([len(s) for s in ref_sents])
            ref_weights = ref_weights / np.sum(ref_weights)
        else:
            raise ValueError(f"Invalid weight type: {self.weight}")
        return hyp_weights, ref_weights

    def compute_ot_cost(
        self,
        hyp_weights: Union[np.ndarray, torch.Tensor],
        ref_weights: Union[np.ndarray, torch.Tensor],
        cost_matrix: Union[np.ndarray, torch.Tensor],
    ) -> Union[float, torch.Tensor]:
        """
        Compute the optimal transport cost using the specified algorithm.

        Args:
            hyp_weights: Weights for hypothesis sentences.
            ref_weights: Weights for reference sentences.
            cost_matrix: Cost matrix for sentence alignments.

        Returns:
            The optimal transport cost.

        Raises:
            ValueError: If the OT algorithm is invalid.
        """
        if (self.ot_alg is None) or (self.ot_alg == "sinkhorn"):
            cost = ot.sinkhorn2(hyp_weights, ref_weights, cost_matrix, 0.1)
        elif self.ot_alg == "exact":
            cost = ot.emd2(hyp_weights, ref_weights, cost_matrix)
        elif self.ot_alg == "greedy":
            cost = ot.bregman.greenkhorn(hyp_weights, ref_weights, cost_matrix, 0.1)
        elif self.ot_alg == "assign":
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            cost = cost_matrix[row_ind, col_ind].mean()
        else:
            raise ValueError(f"Invalid OT algorithm: {self.ot_alg}")
        return cost

    def compute_ot(
        self, hyp_weights: np.ndarray, ref_weights: np.ndarray, cost_matrix: np.ndarray
    ) -> float:
        """
        Compute the optimal transport cost, optionally using GPU acceleration.

        Args:
            hyp_weights: Weights for hypothesis sentences.
            ref_weights: Weights for reference sentences.
            cost_matrix: Cost matrix for sentence alignments.

        Returns:
            The optimal transport cost as a float.
        """
        if self.gpu and (self.ot_alg != "assign"):
            hyp_weights_torch = torch.tensor(
                hyp_weights, dtype=torch.float32, device="cuda"
            )
            ref_weights_torch = torch.tensor(
                ref_weights, dtype=torch.float32, device="cuda"
            )
            cost_matrix_torch = torch.tensor(
                cost_matrix, dtype=torch.float32, device="cuda"
            )
            cost_torch = self.compute_ot_cost(
                hyp_weights_torch, ref_weights_torch, cost_matrix_torch
            )
            cost = cost_torch.cpu().detach().item()
        else:
            cost = self.compute_ot_cost(hyp_weights, ref_weights, cost_matrix)
        return cost

    def set_lang(self, lang: str) -> None:
        """
        Set the language for text processing.

        Args:
            lang: The language code ('en', 'de', 'ja').
        """
        self.lang = lang
        if self.lang == "ja":
            self.nlp = spacy.load("ja_ginza")

    def set_gpu(self, gpu: bool) -> None:
        """
        Set whether to use GPU acceleration.

        Args:
            gpu: Whether to use GPU acceleration.
        """
        self.gpu = gpu
