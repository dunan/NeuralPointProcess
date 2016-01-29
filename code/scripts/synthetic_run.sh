#!/bin/bash

task=mixture

DATA_ROOT=$HOME/Research/NeuralPointProcess/data/synthetic/$task
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess

n_embed=512
H=256
bsize=128
bptt=1
learning_rate=0.001
max_iter=4000
cur_iter=0
T=24
w_scale=0.01
mode=GPU
net=joint
save_dir=$RESULT_ROOT/saved-$task-hidden-$H-embed-$n_embed-bptt-$bptt-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/synthetic \
    -event $DATA_ROOT/event.txt \
    -time $DATA_ROOT/time.txt \
    -lr $learning_rate \
    -device $dev_id \
    -maxe $max_iter \
    -svdir $save_dir \
    -hidden $H \
    -embed $n_embed \
    -T $T \
    -b $bsize \
    -w_scale $w_scale \
    -int_report 100 \
    -int_test 500 \
    -int_save 500 \
    -bptt $bptt \
    -cur_iter $cur_iter \
    -mode $mode \
    -net $net \
    2>&1 | tee $save_dir/log.txt
