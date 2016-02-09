#!/bin/bash

data_root=../../data
sub_folder=real
data_name=taxi

time_file=$data_root/$sub_folder/$data_name/pickup_time.txt
event_file=$data_root/$sub_folder/$data_name/pickup_events.txt
percent=0.2

python split_train_test.py -t $time_file -e $event_file -p $percent -n 100 -s 0
