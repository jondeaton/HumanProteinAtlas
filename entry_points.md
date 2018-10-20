A list of the commands required to run your code from within the Docker container. As a best practice, separate training code from prediction code. For example, if you’re using python, there would be up to three entry points to your code:

python prepare_data.py, which would
Read training data from RAW_DATA_DIR (specified in PATHS.json)
Run any preprocessing steps
Save the cleaned data to CLEAN_DATA_DIR (specified in PATHS.json)
python train.py, which would
Read training data from TRAIN_DATA_CLEAN_PATH (specified in PATHS.json)
Train your model
Save your model to MODEL_DIR (specified in PATHS.json)
python predict.py, which would
Read test data from TEST_DATA_CLEAN_PATH (specified in PATHS.json)
Load your model from MODEL_DIR (specified in PATHS.json)
Use your model to make predictions on new samples
Save your predictions to SUBMISSION_DIR (specified in PATHS.json)