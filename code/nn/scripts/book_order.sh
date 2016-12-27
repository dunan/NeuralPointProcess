#!/bin/bash

task=book_order
prefix_event=event-1
prefix_time=time-1

DATA_ROOT=$HOME/Research/NeuralPointProcess/data/real/$task
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess

hist=0
gru=0
n_embed=128
H=128
bsize=64
bptt=3
learning_rate=0.001
max_iter=400000
cur_iter=0
T=0
w_scale=0.01
mode=CPU
net=time
loss=intensity
time_scale=1
#lambda=27.088494573
lambda=1
save_dir=$RESULT_ROOT/$net-$task-hidden-$H-embed-$n_embed-bptt-$bptt-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/main \
    -history $hist \
    -gru $gru \
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
    -m 0.9 \
    -l2 0 \
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
