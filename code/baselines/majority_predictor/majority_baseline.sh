#!/bin/bash

data_root=../../../data
subfolder=real

task=mimic2

for rr in 1 2 3 4 5; 
do
event_prefix=event-$rr
time_prefix=time-$rr

echo 'predicting ' $task $rr
echo '============= event =============='
python event_majority_baseline.py $data_root/$subfolder/$task/$event_prefix
echo '============= end of event =============='
echo '============= time =============='
python time_mean_baseline.py $data_root/$subfolder/$task/$time_prefix 1
echo '============= end of time =============='
done
