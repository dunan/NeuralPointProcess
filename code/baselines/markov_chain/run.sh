#!/bin/bash

data_root=../../../data
subfolder=real

task=lastfm

event_prefix=event_split_1000
time_prefix=time_split_1000

order=1

echo 'predicting ' $task 'using order=' $order
echo '============= event =============='
python markov_chain_baseline.py $data_root/$subfolder/$task/$event_prefix $order
#python multiorder_markov_baseline.py $data_root/$subfolder/$task/$event_prefix $order
echo '============= end of event =============='
