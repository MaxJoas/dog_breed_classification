import random
import tensorflow as tf
from glob import glob

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications.resnet50 import (ResNet50, decode_predictions,
                                         preprocess_input)
from keras.callbacks import ModelCheckpoint
from keras.layers import (Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import np_utils
from PIL import ImageFile
from sklearn.datasets import load_files
from tqdm import tqdm

from extract_bottleneck_features import *

helper = {'a': [1, 2, 3], 'b': [1, 2, 4]}
df = pd.DataFrame(helper)


# define function to load train, test, and validation datasets

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset('../data/dogImages/train')
valid_files, valid_targets = load_dataset('../data/dogImages/valid')
test_files, test_targets = load_dataset('../data/dogImages/test')

# load list of dog names
dog_names = [item[20:-1]
             for item in sorted(glob("../data/dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' %
      len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.' % len(test_files))


random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("../data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.


# detect if image contains a human face
def test_detectors(human_paths, dog_paths, detector):
    """Helper Function to check the performance of our human/dog detectors

    Args:
        human_paths -(list): list of paths of images containing humans
        dog_paths - (list): list of paths of images containing dogs
        detector - (func): detector function

    Returns:
        human_dict - (dict): counter dict of wrongly/correctly classified humans
        dog_dict - (dict): counter dict of wrongly/correctly classified dogs

    """
    # dict to keep track whether a human is detected or no
    human_dict = {'True': 0, 'False': 0}
    # dict to check if a dog is detected as human
    dog_dict = {'True': 0, 'False': 0}

    for human_img, dog_img in zip(human_paths, dog_paths):
        is_human = detector(human_img)
        is_dog = detector(dog_img)

        human_dict[str(is_human)] += 1
        dog_dict[str(is_dog)] += 1

    return human_dict, dog_dict


# Contemplaiting Perfomrance
perfomance_dicts = test_detectors(human_paths=human_files_short,
                                  dog_paths=dog_files_short,
                                  detector=face_detector)

print("The human detection performance is: {}".format(perfomance_dicts[0]))
print("The dog detection performance is: {}".format(perfomance_dicts[1]))


# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


dog_detector_perfomance = test_detectors(human_files_short,
                                         dog_files_short,
                                         dog_detector)
print("The human detection performance is: {}".format(
    dog_detector_perfomance[0]))
print("The dog detection performance is: {}".format(
    dog_detector_perfomance[1]))


ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load(
    '../data/bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']
valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
model.add(Dense(133, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])


checkpointer = ModelCheckpoint(
    filepath='saved_models/weights.best.Xception.hdf5', verbose=1, save_best_only=True)

model.fit(train_Xception, train_targets,
          validation_data=(valid_Xception, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer])

model.load_weights('saved_models/weights.best.Xception.hdf5')
model.save('./keras_models')

# TODO: Calculate classification accuracy on the test dataset.
preds = [np.argmax(model.predict(np.expand_dims(feature, axis=0)))
         for feature in test_Xception]

# compute accuracy with test data
xception_acc = np.sum(preds == np.argmax(test_targets, axis=1))/len(preds)


def Xception_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    print(bottleneck_feature.shape)
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def classify_dog(path):
    """Classifies a given image wheter it is a dog or human and returns breed of the dog

    Args:
        path - (str): path to the image that should be classified

    Returns:
        dog_breed - (str): Name of the classified dog bred
    """

    # check what kind of image is given
    is_dog = dog_detector(path)
    is_human = face_detector(path)

    # breaks if image is neither dog or human
    if not is_dog and not is_human:
        print("The image could not be identified as dog or human face. \n Please provide a valid image")
        return None

    # detect dog breed
    if is_dog:
        breed = Xception_predict_breed(path)
        # modfiy breed name to be more readable
        breed_friendly = breed.split('.')
        print("Your dog's breed is {}".format(breed_friendly[1]))
        return breed

    if is_human:
        print("This looks more like a human! \n         Anyway, let's see which dog breed looks similar to this human")
        breed = Xception_predict_breed(path)
        # modfiy breed name to be more readable
        breed_friendly = breed.split('.')
        print("This human looks like a {}".format(breed_friendly[1]))
        return breed
