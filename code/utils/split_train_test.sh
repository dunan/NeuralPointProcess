#!/bin/bash

data_root=../../data
sub_folder=synthetic
data_name=mixture-HMM

time_file=$data_root/$sub_folder/$data_name/time-temporal-3.txt
event_file=$data_root/$sub_folder/$data_name/event-temporal-3.txt
percent=0.2

python split_train_test.py -t $time_file -e $event_file -p $percent -n 100 -s 0
