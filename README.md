# Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport

This repository contains the code for the experiments in [Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](https://arxiv.org/abs/2505.23078).

The code is tested on a docker image built on top of [nvidia/pytorch:23.12-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) image.
The code for evaluating the optimal transport-based metrics is in eval-ot-metrics/ directory.

## Installation


```
git clone git@github.com:CyberAgentAILab/mbr-optimal-transport.git
cd mbr-optimal-transport
pip install -r requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh` computes the MBR candidate from the candidates sampled.

### Example on WMT'24 En-Ja

1. Generate candidate outputs randomply sampled from the text generation model.

```
bash ./experiments/sample.sh -d wmt24doc.en-ja -p wmtdoc_en-ja.txt -m sbintuitions/sarashina2.2-1b-instruct-v0.1 -l 2 -z 1 -s 4
```

2. Compute the final MBR output from the generated candidate outputs.

```
bash ./experiments/run_mbr.sh -d wmt24doc.en-ja -m sarashina2.2-1b-instruct-v0.1 -a mbr -i ot-sinkhorn-length-sentbert-ja -l 2 -s 4
```

The results are stored in ./results directory.

## Reference

[Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](https://arxiv.org/abs/2505.23078)

Bibtex:
```
@inproceedings{jinnai-2025-document,
    title = "Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport",
    author = "Jinnai, Yuu",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    publisher = "Association for Computational Linguistics",
}
```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.

## Acknowledgement

The code in eval-ot-metrics/ directory is implemented on top of the code by [Embarrassingly Easy Document-Level MT Metrics](https://github.com/amazon-science/doc-mt-metrics/).
