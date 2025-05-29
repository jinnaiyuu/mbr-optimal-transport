import os
import argparse
from tqdm import tqdm
import json

import numpy as np
import pandas as pd

import boto3

from utility_func import *
from utils import (
    load_dataset,
    load_matrix,
    load_samples_from_file,
    result_dir,
    matrix_dir,
    reward_dir,
)
from parser import get_mbr_parser
from reward_model import load_reward_model


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    prompt = args.prompt

    sample_dir = args.sample_dir

    n_lines = args.n_lines
    start_iter = args.start_iter
    n_samples = args.n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    reward_model_id = args.reward_model
    # TODO: not implemented right now.
    recompute_reward = args.recompute_reward

    compared_dir = args.compared_dir
    c_nsamples = args.c_nsamples

    pairwise = args.pairwise

    get_raw = args.get_raw

    if compared_dir != "None":
        # TODO: variable pairwise is obsolete.
        pairwise = True
    else:
        pairwise = False

    reward_model = load_reward_model(reward_model_id)

    print("reward_model is", reward_model)

    if get_raw is None:
        get_raw = False
    reward_model.set_get_raw(get_raw)
    # if prompt is not None:
    #     reward_model.set_prompt(prompt)

    # if 'PairRM' in reward_model_id:
    #     assert compared_dir != "None"
    #     assert n_samples == 1

    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(
        dataset, ref=True
    )  # TODO: The dataset now has to have a reference.

    client = boto3.client("s3")

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(reward_dir, dataset, model_n), exist_ok=True)
    os.makedirs(os.path.join("../reward_summary", dataset, model_n), exist_ok=True)

    if "-rm" in dataset:
        pass
    else:
        files = sorted(os.listdir(sample_dir))
        filtered_files = load_samples_from_file(
            files, epsilon, topk, topp, True, 0, 0.0
        )
        filtered_files.sort(key=lambda x: int(x.split("_")[0]))
        assert len(filtered_files) > 0

        if compared_dir != "None":
            compared_files = sorted(os.listdir(compared_dir))
            c_filtered_files = load_samples_from_file(
                compared_files, epsilon, topk, topp, True, 0, 0.0
            )
            c_filtered_files.sort(key=lambda x: int(x.split("_")[0]))

        print("first 10 files=", filtered_files[:10])
        print("n_files=", len(filtered_files))

    rows = []

    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break

        if isinstance(src_lines[sample_id], list):
            src_input = src_lines[sample_id][0]["content"]
        else:
            src_input = src_lines[sample_id]

        trg_output = trg_lines[sample_id]
        # src_input = src_lines[sample_id]
        # trg = trg_lines[sample_id]

        if "-rm" in dataset:
            df = pd.DataFrame(trg_lines[sample_id], columns=["text"])
            assert len(df) == 2
        else:
            filename = filtered_files[sample_id]
            assert (
                "{:04}".format(sample_id) in filename
                or "{:05}".format(sample_id) in filename
            )
            df = pd.read_csv(os.path.join(sample_dir, filename))

        assert len(df) >= n_samples
        df = df[:n_samples]
        df.fillna("", inplace=True)
        hyp = list(df.iloc[:]["text"].astype(str))

        if compared_dir != "None":
            # TODO: This only computes the reward of the first hypothesis.
            c_filename = c_filtered_files[sample_id]
            c_df = pd.read_csv(os.path.join(compared_dir, c_filename))
            c_df = c_df[:c_nsamples]
            c_hyps = c_df.iloc[:]["text"].astype(str).tolist()
            scores = [reward_model.get_winratio(src_input, hyp[0], c_hyps)]
        elif pairwise:
            scores = reward_model.get_pairwise_rewards(src_input, hyp)
        # elif prompt is not None:
        #     scores = reward_model.get_rewards(src_input, hyp, trg_output)
        else:
            scores = reward_model.get_rewards(src_input, hyp, trg_output)

        rows.append(scores)

        reward_model_n = os.path.basename(reward_model_id)
        # This is for comparing against different DPO models.
        # if (reward_model_n == 'PairRM') and (os.path.basename(compared_dir) not in model_n):
        #     reward_model_n = 'PairRM_' + os.path.basename(compared_dir)

        if pairwise:
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_{}_pairwise".format(
                sample_id, epsilon, topk, topp, reward_model_n
            )
        elif get_raw:
            # TODO: One can compute the reward and the raw output at the same
            # time. Should we implement in that way?
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_{}_raw".format(
                sample_id, epsilon, topk, topp, reward_model_n
            )
        else:
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}_{}".format(
                sample_id, epsilon, topk, topp, reward_model_n
            )

        outfilepath = os.path.join(reward_dir, dataset, model_n, filename)

        if pairwise:
            df = pd.DataFrame(scores)
        elif get_raw:
            df = pd.DataFrame(scores, columns=[reward_model_id, "raw"])
        else:
            df = pd.DataFrame(scores, columns=[reward_model_id])
        df.to_csv(outfilepath, index=False)
        # df.to_csv(outfilepath, index=False, float_format='%.32f')

        s3_path = os.path.join("fairseq", "mbr", "reward", dataset, model_n, filename)
        client.upload_file(outfilepath, "ailab-jinnai", s3_path)

    if (not pairwise) and (not get_raw):
        df_summary = pd.DataFrame(rows)
        sum_filename = "eps-{:.2f}_topk-{:02d}_topp-{:.2f}_{}".format(
            epsilon, topk, topp, reward_model_n
        )

        sum_path = os.path.join("../reward_summary", dataset, model_n, sum_filename)
        df_summary.to_csv(sum_path, index=False)
        s3_path = os.path.join(
            "fairseq", "mbr", "reward_summary", dataset, model_n, sum_filename
        )
        client.upload_file(sum_path, "ailab-jinnai", s3_path)
