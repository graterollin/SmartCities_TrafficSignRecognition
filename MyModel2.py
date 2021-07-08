import torch 
import torch.nn as nn
import numpy as np
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as T
import pandas as pd
from torchvision.transforms.transforms import Normalize, ToTensor

# Making a labels map of our dataset
labels_map = {
    0: "addedLane",
    1: "curveLeft",
    2: "curveRight",
    3: "dip",
    4: "doNotEnter",
    5: "doNotPass",
    6: "intersection",
    7: "keepRight",
    8: "laneEnds",
    9: "merge",
    10: "noLeftTurn",
    11: "noRightTurn",
    12: "pedestrianCrossing",
    13: "rampSpeedAdvisory20",
    14: "rampSpeedAdvisory35",
    15: "rampSpeedAdvisory40",
    16: "rampSpeedAdvisory45",
    17: "rampSpeedAdvisory50",
    18: "rampSpeedAdvisoryUrdbl",
    19: "rightLaneMustTurn",
    20: "roundabout",
    21: "school",
    22: "schoolSpeedLimit25",
    23: "signalAhead",
    24: "slow",
    25: "speedLimit15",
    26: "speedLimit25",
    27: "speedLimit30",
    28: "speedLimit35",
    29: "speedLimit40",
    30: "speedLimit45",
    31: "speedLimit50",
    32: "speedLimit55",
    33: "speedLimit65",
    34: "speedLimitUrdbl",
    35: "stop",
    36: "stopAhead",
    37: "thruMergeLeft",
    38: "thruMergeRight",
    39: "thruTrafficMergeLeft",
    40: "truckSpeedLimit55",
    41: "turnLeft",
    42: "turnRight",
    43: "yield",
    44: "yieldAhead",
    45: "zoneAhead25",
    46: "zoneAhead45",
}

# Hyperparameters 
num_epochs = 5
num_classes = 3
batch_size = 16
learning_rate = 0.001

MODEL_STORE_PATH = "./models/"

# Load The Data
# Annotation File directories 
training_file = "./LISA_DATA/Training/training_annotations.csv"
validation_file = "./LISA_DATA/Validating/validating_annotations.csv"
testing_file = "./LISA_DATA/Testing/testing_annotations.csv" 

# Traffic Sign Image Directories 
training_images = "./LISA_DATA/Training/training_images"
validating_images = "./LISA_DATA/Validating/validating_images"
testing_images = "./LISA_DATA/Testing/testing_images"

# Creating a custom Dataset for your files 
class CustomImageDataset(Dataset):
    # Run when instantiating the Dataset object 
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # transform and target_transform specify the feature and label transform actions
        self.img_labels = pd.read_csv(annotations_file) # Reads annotations from csv file into labels
        self.img_dir = img_dir
        # Specifies the feature transform actions
        self.transform = transform 
        # Specifies the label transform actions 
        self.target_transform = target_transform
    
    # Returns the number of samples in our dataset
    def __len__(self):
        return len(self.img_labels)

    # Loads and returns a sample from the dataset at the given index 
    def __getitem__(self, idx):
        picture_path = self.img_labels.iloc[idx, 0]
        picture_path = picture_path.split('/')
        picture_path = picture_path[2]
        picture_path = picture_path.split(';')
        picture_path = picture_path[0]
        # Obtains the correct image path
        img_path = os.path.join(self.img_dir, picture_path)
        # Takes image's location on disk, converts that to a tensor using read_image
        image = read_image(img_path)
        # Retrieves the corresponding label from the csv data 
        label = self.img_labels.iloc[idx, 2]
        # Calls the transform functions if applicable 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # Returns the tensor image and corresponding label in a tuple
        return image, label 

# Performs preprocessing on the dataset images 
data_transforms = T.Compose([
    #T.Normalize((0.1307,), (0.3081,)),
    T.Resize((32,32)),
    T.Grayscale(1)
])

# Initializing the custom image dataset
training_data = CustomImageDataset(training_file, training_images, data_transforms)
validation_data = CustomImageDataset(validation_file, validating_images, data_transforms)
testing_data = CustomImageDataset(testing_file, testing_images, data_transforms)

train_loader = DataLoader(training_data, batch_size, shuffle=True)
valid_loader = DataLoader(validation_data, batch_size, shuffle=True)
test_loader = DataLoader(testing_data, batch_size, shuffle=True)

'''
for (X, y) in test_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class EdLeNet(nn.Module):
    def __init__(self):
        super(EdLeNet, self).__init__()
        # Describing the 3 convolutional layers
        self.layer1 = nn.Sequential(
            # TODO: CLARIFY WHAT 30x30x32 is and how to get there!!!!
            # Padding = valid is the same as no padding 
            # 1st param: # of input channels (grayscale, so 1)
            # 2nd param: # of output channels (depth)
            # 3rd param: kernel size (conv filter size)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='valid'),
            # ReLU activation function 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            # Notice that the input is now 32 with the output as 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='valid'),
            nn.ReLU(),
            # TODO: Check the arithmetic on this one, it doesn't end up equally 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Dropout layer to prevent model overfitting
        self.drop_out = nn.Dropout()
        # nn.Linear creates fully convolutional layers
        # The first argument is the number of nodes in the layer,
        # The second argument is the number of nodes in the following layer 
        self.fc1 = nn.Linear(2 * 2 * 128, 120)
        # 3 is the number of classes that we are using at the moment
        self.fc2 = nn.Linear(120, 47)

    # Define how the data flows through these layers when performing a forward pass
    def forward(self, x):
        #print(type(x))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # Reshaping 2x2x128 into 512x1
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out 

# Instantiating the EdLeNet model 
model = EdLeNet()
print(model)

if torch.cuda.is_available():
    model.cuda()

# Loss and optimizer 
# This function combines both a softmax activation and a cross entropy loss function in the same function
criterion = nn.CrossEntropyLoss()
# The first argument passed are the parameters we want the optimizer to train 
# The second argument is the learning rate 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, criterion, optimizer)
    test(test_loader, model, criterion)
print("Done!")