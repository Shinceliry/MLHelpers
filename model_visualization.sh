#!/usr/bin/env bash

modelfile="path to script including target model"
class_name="target model class name"
output_dir="graph output directory path"
output_name="output file name"
model_init_args="{'':''}"  # exsample:"{'config':'path to config file'}"
input_shapes="input tensors shape" # exsample:"1,80,100 1,100,64"

python visualization_models.py \
    --model-file "$modelfile" \
    --model-class "$class_name" \
    --model-init-args "$model_init_args" \
    --obj \
    --input-shapes $input_shapes \
    --output-dir "$output_dir" \
    --output-name "$output_name" \
    --output-format png \
    --rankdir LR \
    --node-color lightblue
