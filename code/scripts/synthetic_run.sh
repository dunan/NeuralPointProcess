#!/bin/bash

task=${1:-mixture}

input_dir="${DATA_ROOT:-$HOME/Research/NeuralPointProcess/data/synthetic}/$task"
RESULT_ROOT=${RESULT_ROOT:-$HOME/scratch/results/NeuralPointProcess}

n_embed=128
H=128
bsize=32
bptt=3
learning_rate=0.01
max_iter=4000
cur_iter=0
w_scale=0.001
mode=CPU
net=joint
save_dir="$RESULT_ROOT/saved-$task-hidden-$H-embed-$n_embed-bptt-$bptt-bsize-$bsize"

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/synthetic \
    -event $input_dir/event.txt \
    -time $input_dir/time.txt \
    -lr $learning_rate \
    -device $dev_id \
    -maxe $max_iter \
    -svdir $save_dir \
    -hidden $H \
    -embed $n_embed \
    -b $bsize \
    -w_scale $w_scale \
    -int_report 100 \
    -int_test 500 \
    -bptt $bptt \
    -cur_iter $cur_iter \
    -mode $mode \
    -net $net \
    2>&1 | tee $save_dir/log.txt
