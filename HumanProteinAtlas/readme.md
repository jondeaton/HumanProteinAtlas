# Human Protein Atlas Data Loader

This python module implements a data loader which allows for access to the Human Protein Atlas
dataset without searching for files 

To use this data loader, first download the Human Protein Atlas Dataset
and make sure that you have the following directory structure

    HumanProteinAtlas
    ├── sample_submission.csv
    ├── test
    ├── train
    └── train.csv
    
Once this is set up, to access the human protein atlas data, begin with

    import HumanProteinAtlas
    dataset = HumanProteinAtlas.Dataset("/path/to/HumanProteinAtlas")
    
this dataset object represents the entire training dataset provided by the human protein
atlas including meta data. This is organized as a collection of samples. You can 
access a single sample like so

    sample_id = dataset.sample_ids[0] # get the first sample ID
    sample = dataset.sample(sample_id)
    
    # or loop through them all like this
    for sample_id in dataset.sample_ids:
        sample = dataset.sample(sample_id)

once you have a sample object you can access the images/labels like so

    sample.red # the red channel image (numpy array)
    sample.multi_channel # red, blue, gree, yellow channels (numpy array)
    sample.labels # python list of localization label (0, 27) inclusive
    
    sample.show() # displays an image of the multi-channel image


Note that access to every image/label is loaded lazily. That is, the entire dataset will
not be loaded into memory up-front but only if you end up touching every single sample.
Also, you can optionally have this loader cache any of the images that are loaded in.


