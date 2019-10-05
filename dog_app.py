from extract_bottleneck_features import extract_Resnet50
from keras.preprocessing import image
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
import cv2
import matplotlib.pyplot as plt
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    name = dog_names[np.argmax(predicted_vector)]
    return name.split('.')[-1]

def load_model():
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=[7,7,2048]))
    Resnet50_model.add(Dropout(0.3))
    Resnet50_model.add(Dense(1024,activation='relu'))
    Resnet50_model.add(Dropout(0.4))
    Resnet50_model.add(Dense(133, activation='softmax'))
    Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
    return Resnet50_model

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def detector(img_path):
    if dog_detector(img_path):
        breed = Resnet50_predict_breed(img_path)
        print('Dog detected, the dog breed is ' + breed)
    elif face_detector(img_path):
        breed = Resnet50_predict_breed(img_path)
        print('Human detected, the similar dog breed is' + breed)
    else:
        print('No dog or human detected')

if __name__ == '__main__':
    image_path = sys.argv[1]
    with open("breeds.txt", "rb") as file:
        dog_names = pickle.load(file)
    Resnet50_model = load_model()
    detector(image_path)
