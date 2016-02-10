#!/bin/bash

data_root=../../../data
subfolder=real

task=so

event_prefix=event
time_prefix=time

for order in 1 2 3; do

echo 'predicting ' $task 'using order=' $order
echo '============= event =============='
python markov_chain_baseline.py $data_root/$subfolder/$task/$event_prefix $order
#python multiorder_markov_baseline.py $data_root/$subfolder/$task/$event_prefix $order
echo '============= end of event =============='

done
