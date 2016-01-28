#!/bin/bash

DATA_ROOT=$HOME/scratch/data/dataset/NeuralPointProcess/lastfm
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess/lastfm

n_hidden=128
bsize=1
learning_rate=0.001
bptt=5
max_iter=40000000
mode=CPU

save_dir=$RESULT_ROOT/saved-hidden-$n_hidden-embed-$n_embed-bptt-$bptt

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/neural_pp_time -time $DATA_ROOT/time.txt -event $DATA_ROOT/event.txt -lr $learning_rate -mode $mode -device $dev_id -maxe $max_iter -svdir $save_dir -hidden $n_hidden -b $bsize -bptt $bptt -int_report 100 -int_test 10000 2>&1 | tee $save_dir/log.txt
