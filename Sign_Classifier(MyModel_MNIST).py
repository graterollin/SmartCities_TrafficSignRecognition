# Author: Andres Graterol 
# Date: 06/17/21
# Models to consider:
# 1. MASK R-CNN with a resnet50 backbone
# 2. YOLOV5 Model 

import os
import pandas as pd
import torch 
from torch import nn 
from torchvision.io import read_image
# Dataloader and Dataset are data primitives that allow us to use pre-loaded datasets as well as your own data
from torch.utils.data import DataLoader # wraps an iterable around the Dataset to enable easy access to the samples
from torch.utils.data import Dataset
from torchvision import datasets # Stores samples and their corresponding labels
from torchvision.transforms import ToTensor, Lambda, Compose 
import torch.onnx as onnx
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Creating a custom Dataset for your files 
class CustomImageDataset(Dataset):
    # Run when instantiating the Dataset object 
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # transform and target_transform specify the feature and label transform actions
        self.img_labels = pd.read_csv(annotations_file) # Reads annotations from csv file into labels
        self.img_dir = img_dir # Sets image directory 
        self.transform = transform 
        self.target_transform = target_transform
    
    # Returns the number of samples in our dataset
    def __len__(self):
        return len(self.img_labels)

    # Loads and returns a sample from the dataset at the given index 
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
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
        return image, label 

# Example code running off of preloaded datasets
# Instiating training and testing data from the FashionMNIST dataset
    
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Iterating and Visualizing the Dataset 
'''
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''

# Dataloader splits the dataset into training, validation, and testing 
# Split should be 70/20/10, respectively
train_dataloader = DataLoader(training_data, batch_size=64) #, shuffle=True)
# Should VALIDATE go here as well????
# validating is needed for the model to optimize its hyperparameters beside the training set
#validate_dataloader = DataLoader(validate_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64) #, shuffle=True)

# Iterate through the DataLoader
# Display image and label
'''
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Get Device for Training 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
'''

# Here we define our neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Create an instance of NeuralNetwork, move it to the device,
# and print its structure 
model = NeuralNetwork() #.to(device)
#print(model)

# Hyperparameters:
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Inside this loop, optimization happens in three steps:
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # OPTIMIZATION
        # Reset the gradients of model parameters. Prevents double-counting
        optimizer.zero_grad()
        # Backpropagation 
        loss.backward()
        # Adjust the parameters by the gradients collected in the backward pass
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches 
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Common loss functions include:
# nn.MSELoss (Mean Square Error) for regression tasks
# nn.NLLLoss (Negative Log Likelihood) for classification 
# nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss
loss_fn = nn.CrossEntropyLoss()
# Check for different optimizing algorithms as they might do a better job!!!!!!!!!!!!!!!!!!
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# We get the prediction probabilities by passing it 
# through an instance of the nn.Softmax module
'''
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Layers --------------------------------------------------------

# Sample minibatch of 3 images of size 28x28
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten 
# Convert each 2D 28x28 image into a contiguous array of 784 pixel values 
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear 
# Applies a linear transformation on the input using its stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# Non-linear activations are what create the complex mappings between the model's inputs and outputs 
# They are applied after linear transformations to introduce nonlinearity,
# helping neural networks learn a wide variety of phenomena
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential 
# The data is passed through all the modules in the same order as defined. 
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax
# Returns logits. They represent the model's predicted probabilities for each class
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Iterate over each parameter, and print its size and a preview of its values 
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
'''
'''
# Saving and Loading model weights
# Saving 
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# Loading 
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights 
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
'''
'''
# Saving and Loading Models with Shapes
# Save 
torch.save(model, 'model.pth')

# Load
model = torch.load('model.pth')
'''