#!/bin/bash

data_root=../../data
sub_folder=synthetic
data_name=exp_temp

time_file=$data_root/$sub_folder/$data_name/time-2.txt
event_file=$data_root/$sub_folder/$data_name/event-2.txt
percent=0.9

python split_train_test.py -t $time_file -e $event_file -p $percent -n 128 -s 1
