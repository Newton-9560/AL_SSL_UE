#!/bin/bash
# run_experiments.sh
# This script runs experiments for each model and dataset combination.

# Define the list of models and datasets (must match the choices in your parse_args() function)
models=("llama3" "opt" "qwen")
datasets=("coqa" "trivia_qa" "sciq")

# Loop over each model and dataset
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running experiment with model: $model and dataset: $dataset"
    
    # Run the Python experiment script with the required arguments
    python hallucination_detection.py \
      --model "$model" \
      --dataset "$dataset" \

  done
done

echo "All experiments completed successfully."
