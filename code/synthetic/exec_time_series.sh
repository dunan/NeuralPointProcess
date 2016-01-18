#!/bin/bash

DATA_ROOT=$HOME/scratch/data/dataset/NeuralPointProcess
GRAPHNN=$HOME/Workspace/cpp/graphnn
RESULT_ROOT=$HOME/scratch/results/MolecularSpace/NeuralPointProcess

H=128
bsize=128
learning_rate=0.001
max_iter=4000
save_dir=$RESULT_ROOT/saved-hidden-$H-bsize-$bsize

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

dev_id=0

$GRAPHNN/build/app/time_series $DATA_ROOT/time.txt -lr $learning_rate -device $dev_id -maxe $max_iter -svdir $save_dir -h $H -b $bsize -int_report 100 -int_test 500 2>&1 | tee $save_dir/log.txt
