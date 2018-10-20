"""
This file needs to follow the provided Python module template for submission, example_submission_predict.py.
It should contain at least a class Model with a method predict. We will use the instantiated class to time the
prediction on the test dataset. The class should accept a string path to the finished model file. The predict
method of the Model class should accept a numpy array with image pixel data and return a set of predicted labels
for the passed in image pixel data. The numpy array should have four dimensions, X, Y, Z and C. X and Y will be
the two sides of a two dimensional image. Z will be the third side of a three dimensional image. In this dataset
there are no three dimensional images, so Z will always be 1. C will be the number of channels per image field
of view. In this dataset there are always four channels. The order of channels in the numpy array is important.
The channel order should be: [red, green, blue, yellow]. The submission_predict.py model should be accessible
inside the docker container when that is run, ie from a Python prompt we should be able to import the Model
class from the module.
"""
