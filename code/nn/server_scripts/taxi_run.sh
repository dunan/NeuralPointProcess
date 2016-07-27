#!/bin/bash

task=taxi
prefix_event=pickup_events
prefix_time=pickup_time

DATA_ROOT=$HOME/data/Research/NeuralPointProcess/data/real/$task
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess

hist=1
gru=1
n_embed=512
H=512
h2=0
bsize=64
bptt=8
learning_rate=0.001
max_iter=4000
cur_iter=0
T=0
w_scale=0.01
mode=GPU
net=time
loss=mse
test_top=1024
time_scale=0.001
lambda=0.0737552764829
unix_str=wHMmd
save_dir=$RESULT_ROOT/$net-$task-gru-$gru-hidden-$H-embed-$n_embed-bptt-$bptt-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=2

./build/main \
    -history $hist \
    -gru $gru \
    -unix_str $unix_str \
    -test_top $test_top \
    -h2 $h2 \
    -unix 1 \
    -t_scale $time_scale \
    -event $DATA_ROOT/$prefix_event \
    -time $DATA_ROOT/$prefix_time \
    -loss $loss \
    -lambda $lambda \
    -lr $learning_rate \
    -device $dev_id \
    -maxe $max_iter \
    -svdir $save_dir \
    -hidden $H \
    -embed $n_embed \
    -save_eval 0 \
    -save_test 1 \
    -T $T \
    -m 0.9 \
    -b $bsize \
    -w_scale $w_scale \
    -int_report 500 \
    -int_test 10000 \
    -int_save 10000 \
    -bptt $bptt \
    -cur_iter $cur_iter \
    -mode $mode \
    -net $net \
    2>&1 | tee $save_dir/log.txt
