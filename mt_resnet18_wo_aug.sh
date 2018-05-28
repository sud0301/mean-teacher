#!/bin/bash

#PBS -l nodes=1:ppn=8:gpus=4
#PBS -l mem=12gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/dense_prediction/cloned/mean-teacher/pytorch/
python -m experiments.cifar10_test   >> mt_resnet18_wo_aug.txt 
