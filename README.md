# CS229-Project
CS229 Final Project


## Usage

Extract features and cluster features

	python -m cell_clustering.cluster \
        --features-dir /path/to/features/directory \
        --model-file  /path/to/model/file \
        --assignments-file /path/to/assignments/file

To train deep model

    python -m deep_model.train

evaluate the deep model

    python -m deep_model.evaluate \
        --save-path /path/to/model_outputs \
        --model model_checkpoint-0.meta \
        --output outputs

To create training, test, and validation sets (this has already been done and should not need to
be done again)

    python -m partitions.create --input ~/Datasets/HumanProteinAtlas --num-test 1500 --num-validation 1500


## Kaggle instructions
Create a README.md file at the top level of the archive. Here is an example file. This file concisely and precisely 
describes the following:

The hardware you used: CPU specs, number of CPU cores, memory, GPU specs, number of GPUs.
OS/platform you used, including version number.
Configuration files (if any) with a description of what the settings mean and
How to train your model
How to make predictions on a new test set.
Important side effects of your code. For example, if your data processing code overwrites the original data.
Key assumptions made by your code. For example, if the outputs folder must be empty when starting a training run.
