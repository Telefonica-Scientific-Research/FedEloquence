#!/bin/bash

model_name="salamandra-2b-instruct"
c=8
dataset="alpaca_cleaned"
work_dir="./"
patience=1
aggregation="FedValLoss"

python ${work_dir}/plot_diagrams.py \
    --model_name "$model_name" \
    --c $c \
    --dataset "$dataset" \
    --work_dir "$work_dir" \
    --patience $patience \
    --aggregation "$aggregation"