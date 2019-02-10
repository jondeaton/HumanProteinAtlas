# HumanProteinAtlas
CS229 Final Project

###### Team members
- Jon Deaton
- Robert Neff
- Luke Sturm


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
