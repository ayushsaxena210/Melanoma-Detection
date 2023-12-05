# Melanoma-Detection

## Table of Contents
* [General Info](#general-info)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Info
This project aims to build a Convolutional Neural Network (CNN) based model for accurately detecting melanoma, a type of skin cancer that can be fatal if not detected early. Melanoma accounts for 75% of skin cancer deaths, and an automated solution for evaluating images and alerting dermatologists about the presence of melanoma can significantly reduce manual effort in diagnosis.

### Business Problem
The primary objective is to develop a CNN model capable of accurately classifying images into one of the nine skin cancer types. Early detection of melanoma through automated image analysis can contribute significantly to saving lives and reducing the manual workload of dermatologists.

### Dataset
The dataset contains images of nine skin cancer types and is divided into train and test subdirectories. The images are loaded and preprocessed using the TensorFlow and Keras libraries.

## Conclusions
1. **Model Overfitting:**
   - The initial model exhibited overfitting, as evidenced by a substantial gap between training accuracy and validation accuracy.
   - Possible causes include insufficient training data, presence of noise in the dataset, or the model's complexity.

2. **Data Augmentation:**
   - To address overfitting, an image data augmentation strategy was implemented using the Augmentor library.
   - Augmented images were added to the training dataset, leading to a more balanced distribution and mitigating overfitting.

3. **Improved Model:**
   - The revised model demonstrated significant improvement, with the gap between training and validation accuracy reduced.
   - The model achieved an accuracy of approximately 85% on the validation set.

## Technologies Used
- TensorFlow
- Keras
- Matplotlib
- Numpy
- Pandas
- Augmentor

## Acknowledgements
- This project was inspired by the need for automated melanoma detection to enhance early diagnosis and improve patient outcomes.
- References: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [Augmentor](https://augmentor.readthedocs.io/en/master/)

## Contact
Created by https://github.com/ayushsaxena210 - feel free to contact me!
