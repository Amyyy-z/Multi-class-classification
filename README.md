# Multi-class-classification

This repository contains the overall implementation guideline, procedures, and partial dataset for multi-classiying thyroid medical images through 10-fold stratified cross-validation. 

## Getting Started 

Before implementing the classification tasks, there are some pre-requisites to comply:

- Computational environment: Tensorflow 2.1.0, Python 3.7 at least, and GPU required
- Data pre-processing completed:
  * If you are working with CT images, segmentation into left and right lobes are required
  * If you are working with ultrasound images, the segmentation step can be skipped
  * All the images must be in the same size 
- Samples of both CT and ultrasound images can be downloaded through [Dataset.zip](https://github.com/Amyyy-z/Multi-class-classification/blob/main/Dataset.zip)

## Implementation Guide

* Prepare the image sets
* Import required libraries
* Import image sets with their labels
* Encode image labels
* Training and Testing splits with stratified cross-validation
* Build the CNN model
* Feed the images and encoded labels into the model for training and testing
* Output the results and the confusion matrix

The detailed step-by-step procedures can be found at [Multi-class classification.py](https://github.com/Amyyy-z/Multi-class-classification/blob/main/Multi-class%20classification.py)

---------------------------

### Thank you!

If you have any further enquires, please contact me through xinyu.zhang@monash.edu
