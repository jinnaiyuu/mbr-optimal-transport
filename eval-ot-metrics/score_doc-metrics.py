# Original Copyright (C) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import argparse
# from bert_score.bert_score.scorer import BERTScorer
from COMET.add_context import add_context, merge_document
# from COMET.comet import download_model, load_from_checkpoint
import json
from mt_metrics_eval.data import EvalSet
import numpy as np
from Prism.prism import MBARTPrism
import torch

import os
import boto3

import logging
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from utility.utility_class import *
from utility.metricx import METRICX
from utility.ot_utility import OTUtility


def select_model(args, device, model_n=None):
    if model_n is None:
        # This is used for the recursive call in the OTUtility class.
        model_n = args.model

    if model_n == "comet":
        model_name = "wmt21-comet-mqm" if not args.qe else "wmt21-comet-qe-mqm"
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        sep_token = model.encoder.tokenizer.sep_token
        if args.doc:
            model.set_document_level()
    elif model_n == "bertscore":
        model = BERTScorer(lang=args.lp.split("-")[-1], rescale_with_baseline=False)
        sep_token = model._tokenizer.sep_token
    elif model_n == "prism":
        model_path = "facebook/mbart-large-50"
        model = MBARTPrism(checkpoint=model_path, src_lang=args.lp.split("-")[-1], tgt_lang=args.lp.split("-")[-1],
                           device=device)
        sep_token = model.tokenizer.sep_token
    elif model_n == "sacrebleu":
        model = SACREBLEU()
        # sep_token is default </s> for sacrebleu
        sep_token = "</s>"
    elif model_n == "BERTSCORE":
        model = BERTSCORE()
        model.set_lang(args.lp.split("-")[-1])
        sep_token = "</s>" # ?
    elif model_n == "metricx":
        model = METRICX("google/metricx-23-xl-v2p0")
        sep_token = "</s>"
    elif model_n == "sentbert":
        model = SENTBERT()
        sep_token = "</s>"
    elif model_n[:3] == "ot-":
        params = model_n.split('-')
        ot_alg = params[1]
        weight = params[2]
        sim_util = params[3]
        lang = params[4]
        
        sim_model = select_model(args, device, sim_util)[0]
        # TODO: sim_model should be a UtilityFunction object. Implement it for bertscore.
        assert isinstance(sim_model, UtilityFunction)
        is_gpu = "cuda:0" == device
        model = OTUtility(sentence_sim=sim_model, ot_alg=ot_alg, weight=weight, lang=lang, gpu=is_gpu)
        sep_token = "</s>"
    else:
        raise ValueError(f"Invalid model: {args.model}")
    return model, sep_token    

def main(args):
    evs = EvalSet(args.campaign, args.lp)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    
    model, sep_token = select_model(args, device)

    logger.info(f"sep_token: {sep_token}")

    scores = {level: {} for level in [args.level]}

    src = evs.src
    orig_ref = evs.all_refs[evs.std_ref]
    
    logger.info(f"len(src): {len(src)}")
    logger.info(f"len(orig_ref): {len(orig_ref)}")
    logger.info(f"src[3]: {src[3]}")
    logger.info(f"orig_ref[3]: {orig_ref[3]}")

    client = boto3.client("s3")


    # keep scores only for the segments that have mqm annotations
    mqm_annot = [True] * len(src)
    if 'no-mqm' in evs.domains:
        for seg_start, seg_end in evs.domains['no-mqm']:
            mqm_annot[seg_start:seg_end] = (seg_end - seg_start) * [False]

    # Used for args.doc and args.ot
    doc_ids = [""] * len(src)
    # get the document ids in a suitable format
    for doc_name, doc_bounds in evs.docs.items():
        doc_start, doc_end = doc_bounds
        doc_ids[doc_start:doc_end] = (doc_end - doc_start) * [doc_name]

    # logger.info(f"doc_ids=", doc_ids)

    # logger.info(f"last doc_name: {doc_name}")
    # logger.info(f"last doc_start: {doc_start}")
    # logger.info(f"last doc_end: {doc_end}")

    if args.doc:
        # add contexts to source and reference texts
        src = add_context(orig_txt=src, context=src, doc_ids=doc_ids, sep_token=sep_token)
        if not args.qe:
            ref = add_context(orig_txt=orig_ref, context=orig_ref, doc_ids=doc_ids, sep_token=sep_token)
            
        logger.info(f"src[3] after add_context: {src[3]}")
        logger.info(f"ref[3] after add_context: {ref[3]}")
    elif args.ot:
        src = merge_document(text=src, doc_ids=doc_ids)
        if not args.qe:
            ref = merge_document(text=orig_ref, doc_ids=doc_ids)
            assert len(src) == len(ref)
        logger.info(f"src[3] after add_context: {src[3]}")
        logger.info(f"ref[3] after add_context: {ref[3]}")
        logger.info(f"len(src) after add_context: {len(src)}")
        logger.info(f"len(ref) after add_context: {len(ref)}")
    else:
        if not args.qe:
            ref = orig_ref

    logger.info(f"len(evs.sys_outputs): {len(evs.sys_outputs)}")
    for sysname, cand in evs.sys_outputs.items():

        # print(f'----Evaluating {sysname}----')
        logger.warning(f'evaluating: {sysname}')
        logger.info(f'cand[3]: {cand[3]}')

        if args.doc:
            # add contexts to hypotheses text
            if args.qe:
                cand = add_context(orig_txt=cand, context=cand, doc_ids=doc_ids, sep_token=sep_token)
            else:
                cand = add_context(orig_txt=cand, context=orig_ref, doc_ids=doc_ids, sep_token=sep_token)
        elif args.ot:
            cand = merge_document(cand, doc_ids)
            assert len(src) == len(cand)
        logger.info(f"cand[3] after context: {cand[3]}")

        if args.model == "comet":
            if args.qe:
                data = [{"src": x, "mt": y} for x, y in zip(src, cand)]
            else:
                data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(src, cand, ref)]
            seg_score, _ = model.predict(data, batch_size=32, gpus=1)
        elif args.model == "bertscore":
            P, R, F1 = model.score(cand, ref, doc=args.doc)
            seg_score = F1.cpu().numpy()
        elif args.model == "prism":
            seg_score = model.score(cand=cand, ref=ref, doc=args.doc, batch_size=2, segment_scores=True)
        elif args.model in ["sacrebleu", "BERTSCORE", "metricx"]:
            seg_score = model.compute_similarity(hyp=cand, ref=ref, src=None)
        elif args.model in ["sentbert"]:
            # TODO: Use mean embedding.
            seg_score = model.compute_similarity(hyp=cand, ref=ref, src=None)
        elif args.model[:3] == "ot-":
            seg_score = model.compute_similarity(hyp=cand, ref=ref, src=None)
        else:
            assert ValueError(f"Invalid model: {args.model}")

        seg_score = [float(x) if mqm_annot[i] else None for i, x in enumerate(seg_score)]
        if args.ot:
            # print("seg_score:", seg_score)
            # print("len(seg_score):", len(seg_score))
            scores[args.level][sysname] = [np.mean(np.array(seg_score))]            
        else:
            scores[args.level][sysname] = [
                np.mean(np.array(seg_score)[mqm_annot]) if args.level == 'sys' else seg_score]
            
        logger.info(f"seg_score[3]: {seg_score[3]}")
        logger.info(f"len(seg_score): {len(seg_score)}")

    if args.save:
        scores_file = "results/{}_{}_{}{}{}_scores.json".format(args.campaign, args.lp, 
                                                          "doc-" if args.doc else "", args.model, "-qe" if args.qe else "")
        with open(scores_file, 'w') as fp:
            json.dump(scores, fp)

        s3_path = os.path.join('fairseq', 'mbr', 'docmt', scores_file.split('/')[-1])
        client.upload_file(scores_file, 'ailab-jinnai', s3_path)


    gold_scores = evs.Scores(args.level, evs.StdHumanScoreName(args.level))
    logger.info(f'gold_scores: {gold_scores}')
    sys_names = set(gold_scores) - evs.human_sys_names
    corr = evs.Correlation(gold_scores, scores[args.level], sys_names)

    if args.level == "sys":
        print(f'system: Pearson={corr.Pearson()[0]:f}')
        pearson_file = "results/{}_{}_{}{}{}_pearson.json".format(args.campaign, args.lp, 
                                                                  "doc-" if args.doc else "", args.model, "-qe" if args.qe else "")
        with open(pearson_file, 'w') as fp:
            pearson_dict = {"Pearson": corr.Pearson()[0]}
            json.dump(pearson_dict, fp)

        s3_path = os.path.join('fairseq', 'mbr', 'docmt', pearson_file.split('/')[-1])
        client.upload_file(pearson_file, 'ailab-jinnai', s3_path)

    else:
        print(f'segment: Kendall-like={corr.KendallLike()[0]:f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce document-level metric scores from the paper.')
    parser.add_argument('--campaign', default='wmt21.news',
                        type=str, help='the wmt campaign to test on')
    parser.add_argument('--lp', default='en-de',
                        type=str, help='the language pair')
    parser.add_argument('--model', default='comet',
                        type=str, help='the model/metric to be tested')
    parser.add_argument('--doc', action="store_true", help='document- or sentence-level comet')
    parser.add_argument('--qe', action="store_true",
                        help='(only for comet) whether evaluation is reference-based or reference-free (qe)')
    
    # OT
    parser.add_argument('--ot', action="store_true", help='evaluate using the whole document at once')
    # parser.add_argument('--assign', action="store_true", help='use assignment-based scoring for bertscore and prism.'
    #                     + 'Not compatible with --doc and --qe and --ot.')
    
    parser.add_argument('--level', required=False, default="sys", choices=["seg", "sys"],
                        help='whether segment-level or system-level scores will be computed')
    parser.add_argument('--save', action="store_true", help='save scores in a file') # save should always be true.

    # OT
    parser.add_argument('--debug', action="store_true", help='print debug information')
    
    args = parser.parse_args()
    args.save = True # save should always be true.

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    main(args)
