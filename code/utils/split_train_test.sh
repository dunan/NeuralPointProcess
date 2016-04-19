#!/bin/bash

data_root=../../data
sub_folder=synthetic
data_name=selfcorrecting

time_file=$data_root/$sub_folder/$data_name/time.txt
event_file=$data_root/$sub_folder/$data_name/event.txt
percent=0.1

python split_train_test.py -t $time_file -e $event_file -p $percent -n 64 -s 1
