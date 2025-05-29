import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from utility_func import *
from utils import (
    load_dataset,
    load_matrix,
    load_samples_from_file,
    result_dir,
    matrix_dir,
)
from parser import get_mbr_parser

from policy.mbr import compute_score_matrix, compute_mbr

from typing import Callable, Optional, Any


def compute_score(
    df: pd.DataFrame,
    d_best: int,
    trg: str,
    compute_evaluate: Callable,
    src: Optional[Any] = None,
) -> float:
    """
    Compute the evaluation score for a selected hypothesis.

    Args:
        df: DataFrame containing the hypotheses.
        d_best: Index of the selected hypothesis.
        trg: Target/reference text.
        compute_evaluate: Function to compute the evaluation score.
        src: Optional source input that may be required by some evaluation functions.

    Returns:
        The evaluation score.
    """
    d_hyp = df.iloc[d_best]["text"]
    d_score = compute_evaluate(d_hyp, trg, src)
    return d_score


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    # TODO: Use sacred instead of argparse.
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model

    sample_dir = args.sample_dir
    matrix_dir = args.matrix_dir

    n_lines = args.n_lines
    start_iter = args.start_iter
    n_samples = args.n_samples

    temperature = args.temperature
    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval

    # Algorithm config
    algorithm = args.algorithm
    recompute_matrix = args.recompute_matrix
    do_sample = args.do_sample

    # Load utility function and evaluation functions
    compute_evaluate, evaluator = load_evaluate(eval_func, None, None)

    # Load dataset
    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    files = sorted(os.listdir(sample_dir))
    print(
        "filter files with epsilon={}, topk={}, topp={}, temperature={}".format(
            epsilon, topk, topp, temperature
        )
    )
    filtered_files = load_samples_from_file(
        files, epsilon, topk, topp, do_sample, 0, 0.0, temperature
    )
    print("TODO: temperature sampling is disabled for now as it causes inconsistency.")

    assert len(filtered_files) > 0

    print("first 10 files=", filtered_files[:10])

    rows = []

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break
        filename = filtered_files[sample_id]
        if not (
            ("{:04}".format(sample_id) in filename)
            or ("{:05}".format(sample_id) in filename)
        ):
            print(
                "Error: sample_id mismatch: sample_id=",
                sample_id,
                "filename=",
                filename,
            )
        assert ("{:04}".format(sample_id) in filename) or (
            "{:05}".format(sample_id) in filename
        )

        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]

        try:
            df = pd.read_csv(os.path.join(sample_dir, filename))
        except BaseException:
            print(
                os.path.join(sample_dir, filename),
                "is not readable with default engine.",
            )
            continue

        assert len(df) >= n_samples
        df = df[:n_samples]
        # TODO: This is needed to remove empty strings. In reality empty
        # strings can be ignored. probably it's better to drop.
        df.fillna("", inplace=True)
        df["text"] = df["text"].astype(str)
        hyp = df.iloc[:]["text"]

        if not recompute_matrix:
            # This makes loading a matrix of size larger
            matrix = load_matrix(
                os.path.join(matrix_dir, dataset, model_n), filename, sim, n_samples
            )
        else:
            matrix = None
        if matrix is None:
            matrix_filename = filename + "_" + sim + "_" + str(n_samples)
            matrix_path = os.path.join(matrix_dir, dataset, model_n, matrix_filename)

            compute_similarity = load_similarity(sim)
            matrix = compute_similarity.compute_score_matrix(hyp, src_input)

            np.savetxt(matrix_path, matrix)

        # MBR: Monte Carlo Estimate
        ed_best = compute_mbr(matrix=matrix)
        ed_score = compute_score(df, ed_best, trg, compute_evaluate, src=src_input)
        row = [sample_id, ed_score, ed_best]
        rows.append(row)

    columns = ["sample_id", "ed_score", "ed_best"]
    postfix = ""

    df = pd.DataFrame(rows, columns=columns)

    if temperature != 1.0:
        filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{:.2f}_{}_{}{}.csv".format(
            dataset,
            model_n,
            n_samples,
            epsilon,
            topk,
            topp,
            temperature,
            sim,
            eval_func,
            postfix,
        )
    else:
        filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(
            dataset, model_n, n_samples, epsilon, topk, topp, sim, eval_func, postfix
        )

    df_path = os.path.join(result_dir, filename)
    df.to_csv(df_path, index=False)
