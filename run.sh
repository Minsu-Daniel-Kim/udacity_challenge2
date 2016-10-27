#!/bin/bash

DATA_DIR="rawdata/driving_dataset"
PARTITION_NUM=15
for i in $(seq 0 3 12);
do
	for j in $(seq 0 2);
	do
    python pipeline/preprocess_pipeline.py -verbose True -totalPartitionNum $PARTITION_NUM -partition $(($i+$j)) --imagePath $DATA_DIR &
   done
done