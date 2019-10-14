[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

In this project, I built a Convolutional Neural Networks (CNN) that classify 1 of the 133 dog breeds in the training dataset. If you supply a picture of your face or your friend's face, the app will also tell you the closet match of a breed.


![Sample Output][image1]

This is the capstone project for my Data scientist nano degree. You can find more about the Nano degree [here](https://www.udacity.com/course/data-scientist-nanodegree--nd025)

## Project Instructions

### Instructions

If you want to train a model yourself, follow the instructions [here](https://github.com/udacity/dog-project.git). If on the other hand, you just want to run the app, follow the following  instructions:

1. Clone the repository and navigate to the downloaded folder.
```
git clone https://github.com/tuanchris/dog-project.git
cd dog-project
```

2. Install required packages

* Linux or Mac:
```
conda create --name dog-project python=3.5
source activate dog-project
pip install -r requirements/requirements.txt
```
* NOTE: Some Mac users may need to install a different version of OpenCV
```
conda install --channel https://conda.anaconda.org/menpo opencv3
```
* Windows:
```
conda create --name dog-project python=3.5
activate dog-project
pip install -r requirements/requirements.txt
```
3. Dog breed/human detection
```
python dog_app.py /path/to/image
```

## Evaluation

This model was able to achieve 81% accuracy on the test set.  

## Further improvements
Here are some further improvements for the model
* Augment training data to prevent overfit
* Tune model parameters to improve accuracy
* Try other models
