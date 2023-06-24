# CV: Face Recognition
Members: **Jiaqin Yan, Siyi Wang, Xi Chen**

## Introduction

We are going to perform a face recogition. The major task is given a face image, find all the face from the same person in image database.
Actually it is multi-label classification problem, because we need to treat all the face image from the same person as one label. There are two major challenges:

1. **Similarity of faces.**
    Face has features that is easy to distinguish from the background, but distinguish different people's face is hard.

2. **Different expressions of faces.**
   Recognising face images from the same person with different expressions is really hard.

## Environment setup
Python version: 3.7
Package needed: **deepface, keras, numpy**

You can use the following commands to setup your environment:
```
conda create --name CVProj python=2.7
conda install deepface
conda install numpy
conda install keras
conda activate CVProj
```
## Usage

### Train and evaluate all models
... explain the works ...

run commands as follows:
```
TODO
```

### Use trained models to recognize faces
... explain the works ...

run commands as follows:
```
TODO
```