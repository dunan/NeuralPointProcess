#!/bin/bash

task=lastfm
f_event=event.txt
f_time=time.txt

DATA_ROOT=$HOME/Research/NeuralPointProcess/data/real/$task
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess

n_embed=128
H=128
bsize=64
bptt=2
learning_rate=0.0001
max_iter=4000
cur_iter=0
test_pct=0.1
T=0
w_scale=0.01
mode=CPU
net=joint
save_dir=$RESULT_ROOT/$net-$task-hidden-$H-embed-$n_embed-bptt-$bptt-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/main \
    -test_pct $test_pct \
    -event $DATA_ROOT/$f_event \
    -time $DATA_ROOT/$f_time \
    -lr $learning_rate \
    -device $dev_id \
    -maxe $max_iter \
    -svdir $save_dir \
    -hidden $H \
    -embed $n_embed \
    -save_eval 0 \
    -T $T \
    -b $bsize \
    -w_scale $w_scale \
    -int_report 500 \
    -int_test 2500 \
    -int_save 2500 \
    -bptt $bptt \
    -cur_iter $cur_iter \
    -mode $mode \
    -net $net \
    2>&1 | tee $save_dir/log.txt
