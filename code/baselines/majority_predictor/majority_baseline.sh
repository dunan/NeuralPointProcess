#!/bin/bash

data_root=../../../data
subfolder=real

task=taxi

event_prefix=pickup_events
time_prefix=pickup_time

echo 'predicting ' $task
echo '============= event =============='
python event_majority_baseline.py $data_root/$subfolder/$task/$event_prefix
echo '============= end of event =============='
echo '============= time =============='
python time_mean_baseline.py $data_root/$subfolder/$task/$time_prefix 0.001
echo '============= end of time =============='
