## TL;DR
*  Built a CNN that classify dog breeds with an accurcay of 85%
*  Try it out [here](https://dogbreedclassification.herokuapp.com)


## Project Motivation

This project was created as part of my Udacity Data Scienc Nanodegree. The goal was to build a convolutional neural network (CNN) to classify dog breeds. Since my past expierence had a focus on classical statistical/machine learning I was curious to work with artifical neural networks.

This project has two main parts:
1. A Notebook that documents my whole work process
2. A Web app, where users can upload images of their dogs and classify the breed! Try it out [here](https://dogbreedclassification.herokuapp.com)

## Requirements
The required libraries can be found in `requirements.txt`

## File Structure
*  `dog_app.py` Script that loads data, builds, trains and save the CNN model
*  `dog_app.ipynb` Jupyter Notebook (partly filled by Udacity) that documents the working process of creating the CNN model
* `extract_bottleneck_features.py` Helper script to work with popular CNN architectures e.g. Resnet50, VGG19, InceptionV3...

## Data Availability

The image data comes from [ImageNet](http://www.image-net.org) and was partly prepared by Udacity. Hence I don't share the data on github. However, I'd like to  present the file structure of the data. All data was placed in a data directory, which was one level above this git directory.
The `test`, `train` and `valid` directories have all the same structure and containing jpg files provided by Udacity from ImageNet.
The `lfw` direcotry contains images of human faces. This data was used to detect wheter the image is a human or a dog.
The `bottleneck_features` contains pretrained weights by Xception and VGG16. In `own_data` I saved personal pictures to test the model
In structure inside the data folder looks as follows:


```
data  
│
└───dogImages
│   │
│   └───test
│     │
│     └───1.breed01
│         │   breed01_1.jpg
│         │   breed01_2.jpg
│         │   ...
│     └───1.breed02
│         │   breed02_1.jpg
│         │   breed02_2.jpg
│         │   ...
│     └─── ...
│         │   ... 
│   └───train
│     └─── ...
│         │   ... 
│   └───valid
│     └─── ...
│         │   ... 
└───lfw
|    │   
|    └───human01
│       │   human01.jpg
|    └───human02
│       │   human02.jpg
│   └─── ...
│       │   ... 
└───bottleneck_features
|    │   
|    └───DogXceptionData.npz
|    └───DogVGG16Data.npz
|___own_test
|    │   
|    └───own_img.jpg
```

## Web App
In order to make the model available to a less tech-savvy audience, I've created a Flask Web App and deployed it to heroku.
Users can upload images to the website. If it is an image of a dog, the website classifies the dog's breed. If it is an image of a human, the website detects the image as a human and finds a dog breed that ressembles the human.
*  Try it out [here](https://dogbreedclassification.herokuapp.com)
*  Find the code [here](https://github.com/MaxJoas/dogBreedWebApp)



## Acknowledgements
Udacity provided and structured the images from [ImageNet](http://www.image-net.org) as well as a structure for the notebook.
ResNet 50, Xception and VGG16 are prebuilt CNN architectures that have been used.
