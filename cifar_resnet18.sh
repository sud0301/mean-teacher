#!/bin/bash

#PBS -l nodes=1:ppn=8:gpus=4
#PBS -l mem=12gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/dense_prediction/cloned/mean-teacher/pytorch/
python main.py \
    --dataset cifar10 \
    --labels data-local/labels/cifar10/1000_balanced_labels/00.txt \
    --arch cifar_shakeshake26 \
    --consistency 100.0 \
    --consistency-rampup 5 \
    --labeled-batch-size 62 \
    --epochs 180 \
    --lr-rampdown-epochs 210  >> mt_resnet18.txt 
