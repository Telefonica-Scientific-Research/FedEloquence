#!/bin/bash

model_name="salamandra-2b-instruct"
c=8
dataset="alpaca_cleaned"
work_dir="./"
patience=1
early_stop="LDES"
fl_method="FedAvg"
client_language_composition="all"

python "$work_dir/plot_diagrams.py" \
    --model_name "$model_name" \
    --c "$c" \
    --dataset "$dataset" \
    --work_dir "$work_dir" \
    --patience "$patience" \
    --early_stop "$early_stop" \
    --fl_method "$fl_method" \
    --client_language_composition "$client_language_composition"
