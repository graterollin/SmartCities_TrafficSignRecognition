# Author: Andres Graterol
# Based on the Model Architecture described in:
# https://towardsdatascience.com/recognizing-traffic-signs-with-over-98-accuracy-using-deep-learning-86737aedc2ab

# Kernel size 3x3: 
# Input Image: (32, 32, 1) (Grayscale image)
# Conv Layer 1 - ReLU (30, 30, 32) (Depth of 32) (activation function)
# 2x2 Max pooling operation: Max Pool (15, 15, 32)
# Conv Layer 2 - ReLU (13, 13, 64)
# Max Pool (6, 6, 64)
# Conv Layer 3 - ReLU (4, 4, 128)
# Max Pool (2, 2, 128)

# 3 Fully Connected Layers:
'''
Final layer produces x results (the total number of posssible labels)
computed using the SoftMax activation function
'''
# Network is trained using mini-batch stochastic gradient descent with the ADAM optimizer
  
# The ModelConfig contains information about the model such as:
# Input format: [32, 32, 1] for grayscale 

'''
convolutional layers config [filter size, start depth, number of layers]
filter size = 3 or 5 (depending on kernel)
start depth = 32 
number of layers = 3 or 2 (depending on kernel)
''' 
# Fully connected layers dimensions: [120, 84]
# Number of classes 

'''
dropout keep percentage values [p-conv, p-fc]
p-conv: 0.75 probability of keeping weight in convolutional layer 
p-fc: 0.5 probability of keeping weight in fully connected layer 
'''
# The ModelExecutor is responsible for training, evaluating, predicting, and producing visualizations of our activation maps 
# ---------------------------------------------------------------------------------------------------------------------------------
# Load pickled data
import pickle
import pandas as pd
import matplotlib as plt

# STEP 0: Load The Data
training_file = "./LISA_DATA/Training/training_annotations.csv"
validation_file = "./LISA_DATA/Validating/validating_annotations.csv"
testing_file = "./LISA_DATA/Testing/testing_annotations.csv" 

'''
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
'''

train = pd.read_csv(training_file)
valid = pd.read_csv(validation_file)
test = pd.read_csv(testing_file)

'''
# Pickled Data: 
# 'features': 4D array containing raw pixel data of the traffic sign images (num examples, width, height, channels)
# 'labels': 1D array containing the label/class id of the traffic sign
# 'sizes' is a list containing (width, height)
# 'coords' is a list containing tuples representing coordinates of a bounding box around the sign 

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
'''

print(train.shape)
print(valid.shape)
print(test.shape)

# STEP 1: Dataset Summary & Exploration 
# Data Summary:

# Number of training examples 
n_train = train.shape[0]

# Number of validation examples 
n_validation = valid.shape[0]

# Number of testing examples
n_test = test.shape[0]

# What's the shape of a traffic sign image 
# image_shape = 

# How many unique classes/labels there are in the dataset 
n_classes = 3

print("Number of training examples = ", n_train)
print("Numer of validating = ", n_validation)
print("Number of testing examples = ", n_test)
#print("Image data shape = ", image_shape)
print("Number of classes = ", n_classes)

# Exploratory visualization of the dataset 