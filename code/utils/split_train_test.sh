#!/bin/bash

data_root=../../data
sub_folder=real
data_name=book_order

time_file=$data_root/$sub_folder/$data_name/time.txt
event_file=$data_root/$sub_folder/$data_name/event.txt
percent=0.2

for rr in 2 3 4 5; do

python split_train_test.py -t $time_file -e $event_file -p $percent -n 100 -s 1 -r $rr

done
