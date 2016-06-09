#!/bin/bash

data_root=../../data
sub_folder=real
data_name=mimic2

time_file=$data_root/$sub_folder/$data_name/time.txt
event_file=$data_root/$sub_folder/$data_name/event.txt
percent=0.1

for rr in 2 3 4 5; do

python split_train_test.py -t $time_file -e $event_file -p $percent -s 0 -r $rr

done
