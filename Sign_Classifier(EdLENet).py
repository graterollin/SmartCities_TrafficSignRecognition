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
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch 
from torch import nn
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T

# NOTE: img_dir was removed from init and therefore getitem since 
# the image path is included inside of the .csv annotation file 
# Creating a custom Dataset for your files 
class CustomImageDataset(Dataset):
    # Run when instantiating the Dataset object 
    def __init__(self, annotations_file, transform=None, target_transform=None):
        # transform and target_transform specify the feature and label transform actions
        self.img_labels = pd.read_csv(annotations_file) # Reads annotations from csv file into labels
        self.transform = transform 
        self.target_transform = target_transform
    
    # Returns the number of samples in our dataset
    def __len__(self):
        return len(self.img_labels)

    # Loads and returns a sample from the dataset at the given index 
    def __getitem__(self, idx):
        image_path = self.img_labels.iloc[idx, 0]
        image_path = image_path.split(';')
        img_path = image_path[0]
        img_path = os.path.join("./LISA_DATA", img_path)
        # Takes image's location on disk, converts that to a tensor using read_image
        image = read_image(img_path)
        # Retrieves the corresponding label from the csv data 
        label = self.img_labels.iloc[idx, 1]
        # Calls the transform functions if applicable 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # Returns the tensor image and corresponding label in a tuple
        #print(image_path)
        return image, label 

# Performs preprocessing on the dataset images 
data_transforms = T.Compose([
    T.Resize((32,32)),
    T.Grayscale(1)
])

# STEP 0: Load The Data
training_file = "./LISA_DATA/Training/training_annotations.csv"
validation_file = "./LISA_DATA/Validating/validating_annotations.csv"
testing_file = "./LISA_DATA/Testing/testing_annotations.csv" 

training_data = CustomImageDataset(training_file, data_transforms)
validation_data = CustomImageDataset(validation_file, data_transforms)
testing_data = CustomImageDataset(testing_file, data_transforms)

# Using __getitem__ in a loop of our dataset(annotations.csv length)
print(training_data.__getitem__(0))
print("\n")

# Each element in the dataloader iterable 
# will return a batch of 64 features and labels 
batch_size = 64

# Loading data using dataloaders
# Reshuffle the data at every epoch to reduce model overfitting 
# If using CUDA, num_workers should be set to 1 and pin_memory to True  
train_dataloader = DataLoader(training_data, batch_size=batch_size)
valid_dataloader = DataLoader(validation_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)

for (X, y) in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    #print("\nShape of y: ", y.shape, y.dtype)
    break

# Exploratory visualization of the dataset 
# Iterate through the DataLoader
# Display image and label
# Array of features & labels will be [0..63] due to batch size
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

'''
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''

'''
# Loading using pickle (not suggested)
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
'''

'''
# Loading using pandas 
train = pd.read_csv(training_file)
valid = pd.read_csv(validation_file)
test = pd.read_csv(testing_file)

print(train.shape)
print(valid.shape)
print(test.shape)
'''

# Pickled Data: 
# 'features': 4D array containing raw pixel data of the traffic sign images (num examples, width, height, channels)
# 'labels': 1D array containing the label/class id of the traffic sign
# 'sizes' is a list containing (width, height)
# 'coords' is a list containing tuples representing coordinates of a bounding box around the sign 
'''
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Features shape: ", X_train.shape)
print("Leatures shape: ", y_train.shape)
'''

# STEP 1: Dataset Summary & Exploration 
# Data Summary:

'''
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


# Get Device for Training 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Step 2: Design and Test a Model Architecture 
'''