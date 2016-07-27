#!/bin/bash

task=book_order
prefix_event=event
prefix_time=time

DATA_ROOT=$HOME/data/Research/NeuralPointProcess/data/real/$task
RESULT_ROOT=$HOME/scratch/results/NeuralPointProcess

gru=$1
hist=$gru
n_embed=$2
H=$3
h2=$4
bsize=$5
bptt=$6
learning_rate=0.001
max_iter=400
cur_iter=0
T=0
w_scale=0.01
mode=CPU
net=event
loss=mse
time_scale=1
lambda=27.088494573
save_dir=$RESULT_ROOT/$net-$task-gru-$gru-hidden-$H-h2-$h2-embed-$n_embed-bptt-$bptt-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

../build/main \
    -h2 $h2 \
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
