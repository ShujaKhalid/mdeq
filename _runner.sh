#!/bin/sh

python ./tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml --lbd=0.0
python ./tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml --lbd=0.2
python ./tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml --lbd=0.4
python ./tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml --lbd=0.6
python ./tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml --lbd=0.8
python ./tools/cls_train.py --cfg experiments/cifar/cls_mdeq_TINY.yaml --lbd=1.0
