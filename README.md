# PyTorch Implementation of KPRN

## Introduction

This repository is Pytorch implementation of [Knowledge-guided Pairwise Reconstruction Network for Weakly Supervised Referring Expression Grounding](https://arxiv.org/pdf/1909.02860.pdf) in ACM MM 2019.
Check our [paper](https://arxiv.org/pdf/1909.02860.pdf) for more details.


## Prerequisites

* Python 3.5
* Pytorch 0.4.1
* CUDA 8.0

## Installation

1. Please refer to [MattNet](https://github.com/lichengunc/MAttNet) to install [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn), [REFER](https://github.com/lichengunc/refer) and [refer-parser2](https://github.com/lichengunc/refer-parser2).
Follow Step 1 & 2 in Training to prepare the data and features.

2. Calculate semantic similarity as supervision infotmation.

* Download Glove word embedding.
```bash
cache/word_embedding/download_embed_matrix.sh
```

* Generate semantic similarity and word embedding file.
```bash
python tools/prepro_sim.py --dataset ${DATASET} --splitBy ${SPLITBY}
```

## Training

Train KPRN with ground-truth annotation:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py --dataset ${DATASET} --splitBy ${SPLITBY} --exp_id ${EXP_ID} --sub_filter_type ${SUBJECT_FILTER_TYPE} --sub_filter_thr ${SUBJECT_FILTER_THRESHOLD}
```

## Evaluation

Evaluate KPRN with ground-truth annotation:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval.py --dataset ${DATASET} --splitBy ${SPLITBY} --split ${SPLIT} --id ${EXP_ID}
```


## Citation

    @inproceedings{lxj2019kprn,
      title={Knowledge-guided Pairwise Reconstruction Network for Weakly Supervised Referring Expression Grounding},
      author={Xuejing Liu, Liang Li, Shuhui Wang, Zheng-Jun Zha, Li Su, and Qingming Huang},
      booktitle={ACM MM},
      year={2019}
    }


## Acknowledgement

Thanks for the work of [Licheng Yu](http://cs.unc.edu/~licheng/). Our code is based on the implementation of [MattNet](https://github.com/lichengunc/MAttNet).

## Authorship

This project is maintained by [Xuejing Liu](https://gingl.github.io/).
