# TL_Classifier
Development of a Traffic Light Classifier for Udacity Capstone project

Using only 500 images to train a neural network to identify the colour of the traffic light within the picture.
The training heavily banks on creating artificial data through image augmentation ([see related file](scripts/preprocessing.py)). The model was created in Tensorflow ([see file](scripts/train_model.py))
and is saved within [this folder](/scripts/model).

The said model performed with 100% accuracy on a random test set of images. Please feel free to try it out and your comments are welcome!
