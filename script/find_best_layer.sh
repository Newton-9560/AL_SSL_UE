#!/bin/bash
# run_experiments.sh
# This script runs experiments for each model and dataset combination.

# Define the list of models and datasets (must match the choices in your parse_args() function)
# models=("llama3" "opt" "qwen")
# datasets=("coqa" "trivia_qa" "sciq")

models=("mistral")
datasets=("coqa" "trivia_qa" "simple_qa" "truthful_qa")

# Loop over each model and dataset
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running experiment with model: $model and dataset: $dataset"
    
    # Run the Python experiment script with the required arguments
    python find_best_layer.py \
      --model "$model" \
      --dataset "$dataset" \

  done
done

echo "All experiments completed successfully."


# CUDA_VISIBLE_DEVICES=2 python generate_dataset.py \
#   --model mistral \
#   --model_size 7 \
#   --dataset simple_qa

# CUDA_VISIBLE_DEVICES=3 python generate_dataset.py \
#   --model qwen \
#   --dataset ambig_qa

# #! SERVER
# CUDA_VISIBLE_DEVICES=1 python generate_dataset.py \
#   --model mistral \
#   --model_size 7 \
#   --dataset coqa \
#   --num_samples 800

# CUDA_VISIBLE_DEVICES=2 python generate_dataset.py \
#   --model mistral \
#   --model_size 7 \
#   --dataset trivia_qa \
#   --num_samples 4000


# CUDA_VISIBLE_DEVICES=3 python generate_dataset.py \
#   --model mistral \
#   --model_size 7 \
#   --dataset simple_qa \
#   --num_samples 4000


# # CUDA_VISIBLE_DEVICES=0 python generate_dataset.py \
# #   --model mistral \
# #   --model_size 7 \
# #   --dataset truthful_qa \
# #   --num_samples -1


# # CUDA_VISIBLE_DEVICES=1 python generate_dataset.py \
# #   --model mistral \
# #   --model_size 7 \
# #   --dataset tydiqa \
# #   --num_samples -1


# CUDA_VISIBLE_DEVICES=2 python generate_dataset.py \
#   --model mistral \
#   --model_size 7 \
#   --dataset simple_qa \
#   --num_samples 4000


# CUDA_VISIBLE_DEVICES=6 python generate_dataset.py \
#   --model mistral \
#   --model_size 7 \
#   --dataset ambig_qa \
#   --num_samples 4000


# CUDA_VISIBLE_DEVICES=7 python generate_dataset.py \
#   --model mistral \
#   --model_size 7 \
#   --dataset sciq \
#   --num_samples 4000
