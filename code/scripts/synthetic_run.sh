#!/bin/bash

DATA_ROOT=$HOME/Research/LSTMPointProcess/data/synthetic/hawkes
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess

H=128
bsize=256
bptt=8
learning_rate=0.0001
max_iter=4000
cur_iter=0
save_dir=$RESULT_ROOT/saved-hidden-$H-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/synthetic -event $DATA_ROOT/event.txt -time $DATA_ROOT/time.txt -lr $learning_rate -device $dev_id -maxe $max_iter -svdir $save_dir -hidden $H -b $bsize -int_report 100 -int_test 500 -bptt $bptt -cur_iter $cur_iter 2>&1 | tee $save_dir/log.txt
