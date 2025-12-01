#!/bin/bash

# Path to your xl-vlms repository
cd ~/xai/xl-vlms

model_name=llava-hf/llava-1.5-7b-hf
#model_name=allenai/Molmo-7B-D-0924
#model_name=HuggingFaceM4/idefics2-8b
#model_name=Qwen/Qwen2-VL-7B-Instruct

# Need to specify analysis type. This will decompose features, and extract both text, visual grounding
analysis_name=decompose_activations_text_grounding_image_grounding

# Specify path where you have saved features on training data
# saved_features_path=/home/ytllam/xai/xl-vlms/out/features/save_hidden_states_for_token_of_interest_llava_dog_generation_split_train.pth
# saved_features_path=/home/ytllam/xai/xl-vlms/out/features/save_hidden_states_for_token_of_interest_llava_dog_generation_split_train_single.pth
# saved_features_path=/home/ytllam/xai/xl-vlms/out/features/save_hidden_states_for_token_of_interest_llava_dog_generation_split_train_single_old.pth
# saved_features_path=/home/ytllam/xai/xl-vlms/out/features/save_hidden_states_for_token_of_interest_llava_dog_generation_split_train_batched.pth
# saved_features_path=/home/ytllam/xai/xl-vlms/out/features/save_hidden_states_for_token_of_interest_llava_dog_generation_split_train_batched_old.pth
saved_features_path=/home/ytllam/xai/xl-vlms/out/features/save_hidden_states_for_token_of_interest_llava_dog_generation_split_train_batched_new.pth

# Where to store details about extracted concepts. Default directory is results/
# results_filename=results_dog_train
# results_filename=results_dog_train_single
# results_filename=results_dog_train_batched
# results_filename=results_dog_train_single_old
# results_filename=results_dog_train_batched_old
results_filename=results_dog_train_batched_new

# Specify the specific feature module you want to decompose

# LLaVA-v1.5
feature_module=language_model.model.norm
#feature_module=language_model.model.layers.30
feature_module=language_model.model.layers.31

# Molmo-7B
#feature_module=model.transformer.blocks.27
#feature_module=model.transformer.ln_f

# Idefics2-8B
#feature_module=model.text_model.norm

# Qwen2-VL-7B
#feature_module=model.norm

decomposition=snmf # Current options: snmf, kmeans, pca, simple
n_concepts=20 # Size of dictionary learnt i.e. number of concepts

python src/analyse_features.py \
--model_name $model_name \
--analysis_name $analysis_name \
--features_path $saved_features_path \
--module_to_decompose $feature_module \
--num_concepts $n_concepts \
--decomposition_method $decomposition \
--save_filename $results_filename