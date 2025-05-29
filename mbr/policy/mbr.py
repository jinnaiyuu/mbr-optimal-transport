from typing import List, Optional, Union, Callable, Any

import numpy as np


def compute_score_matrix(
    samples: List[str],
    score_function: Callable[[List[str], List[str], Any], List[float]],
    src_input: Optional[Any] = None,
) -> np.ndarray:
    """
    Compute a similarity matrix between all pairs of samples.

    Args:
        samples: A list of text samples to compare.
        score_function: A function that computes similarity scores between hypotheses and references.
        src_input: Optional source input that may be required by some similarity functions.

    Returns:
        A numpy array representing the similarity matrix where each element [i,j]
        is the similarity score between samples[i] and samples[j].

    Note:
        TODO: add param ref_samples to compute the score between two different samples.
    """
    n_samples = len(samples)
    scores = []
    for i in range(n_samples):
        score = score_function(
            hyp=np.array([samples[i]] * n_samples), ref=samples, src=src_input
        )
        scores.append(score)
    return np.array(scores)


def compute_mbr(
    hyp: Optional[List[str]] = None,
    compute_similatiy: Optional[Callable] = None,
    matrix: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    src: Optional[Any] = None,
    incremental: bool = False,
) -> Union[int, List[int]]:
    """
    Compute the Minimum Bayes Risk (MBR) decoding result.

    Args:
        hyp: List of hypothesis strings.
        compute_similatiy: Function to compute similarity between hypotheses.
        matrix: Pre-computed similarity matrix. If provided, hyp and compute_similarity are not needed.
        weights: Optional weights for each hypothesis. If not provided, uniform weights are used.
        src: Optional source input that may be required by some similarity functions.
        incremental: If True, returns the best hypothesis at each step of processing.

    Returns:
        If incremental is False, returns the index of the best hypothesis.
        If incremental is True, returns a list of indices of the best hypothesis at each step.

    Note:
        Either (hyp and compute_similatiy) or matrix must be provided.
    """
    assert (compute_similatiy is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, compute_similarity, [src] * len(hyp))

    if weights is not None:
        mbr_scores = matrix @ np.transpose(weights)
    else:
        mbr_scores = np.sum(matrix, axis=1)

    if incremental:
        best_hyp = -1
        best_score = -np.inf
        bests = []
        for i in range(mbr_scores.shape[0]):
            if mbr_scores[i] > best_score:
                best_hyp = i
                best_score = mbr_scores[i]
            assert best_hyp >= 0
            bests.append(best_hyp)
        return bests  # List of hypothesis indices.
    else:
        best_hyp = np.argmax(mbr_scores)

        assert best_hyp >= 0
        return best_hyp
