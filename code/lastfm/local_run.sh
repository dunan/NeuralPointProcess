#!/bin/bash

DATA_ROOT=$HOME/scratch/data/dataset/NeuralPointProcess/lastfm
GRAPHNN=$HOME/Workspace/cpp/graphnn
RESULT_ROOT=$HOME/scratch/results/MolecularSpace/NeuralPointProcess/lastfm

n_hidden=256
n_embed=128
bsize=16
learning_rate=0.001
bptt=3
max_iter=4000
mode=CPU

save_dir=$RESULT_ROOT/saved-hidden-$n_hidden-embed-$n_embed-bptt-$bptt

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

./build/neural_pointprocess -time $DATA_ROOT/time.txt -event $DATA_ROOT/event.txt -lr $learning_rate -mode $mode -device $dev_id -maxe $max_iter -svdir $save_dir -hidden $n_hidden -embed $n_embed -b $bsize -int_report 100 -int_test 500 2>&1 | tee $save_dir/log.txt
