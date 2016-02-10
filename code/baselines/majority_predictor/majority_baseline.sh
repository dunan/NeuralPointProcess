#!/bin/bash

data_root=../../../data
subfolder=real

task=lastfm

event_prefix=event_split_1000
time_prefix=time_split_1000

echo 'predicting ' $task
echo '============= event =============='
python event_majority_baseline.py $data_root/$subfolder/$task/$event_prefix
echo '============= end of event =============='
echo '============= time =============='
python time_mean_baseline.py $data_root/$subfolder/$task/$time_prefix 1
echo '============= end of time =============='
