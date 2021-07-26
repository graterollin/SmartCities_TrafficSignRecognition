import torch 
import torch.nn as nn
import numpy as np
import os
from functools import partial
from torch.optim import optimizer
from torch.utils.data.dataset import random_split
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms.transforms import Grayscale, Normalize, ToTensor
from csv import DictReader
from csv import reader
import atexit
import inspect
import signal
import time
import sys
import subprocess
import ray
ray.init(include_dashboard=False)
from ray import tune
from ray.tune import CLIReporter 
from ray.tune.schedulers import ASHAScheduler

# Making a labels map (dictionary - keys, values) of our dataset
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
epochs = 10
num_classes = 47
batch_size = 16
learning_rate = 0.001

MODEL_STORE_PATH = "./models/"

'''
CREATE_SUSPENDED = 0x00000004  # from Windows headers
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

process = subprocess.Popen(
    creationflags=(CREATE_SUSPENDED if True else 0) | CREATE_BREAKAWAY_FROM_JOB)
'''

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

# Needs config data 
def load_data():
    # Performs preprocessing on the dataset images (training) 
    data_transforms = T.Compose([
        # TODO: Add image normalization transform to see how it effects model accuracy
        T.Resize((32,32)),
        # CHECK THIS NORMALIZATION
        T.Grayscale(1),
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    ])

    # This compose consists of image manipulation - Vertical flip, Horizontal Flip, Rotation 
    basic_data_augment = T.Compose([
        T.Resize((32,32)),
        T.Grayscale(1),
        # The following transformations occur at random 
        T.RandomRotation(degrees=(0, 180)),
        # Horizontal and Vertical flips have 50% probability of happening
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5)
    ])

    # This compose does NOT turn images to grayscale before testing 
    # This compose consists of Random Crop and Random affine (just like in TDS article)
    visual_data_augment = T.Compose([
        T.Resize((32, 32)),
        T.Grayscale(1),
        T.RandomResizedCrop(size=(32, 32)),
        #T.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5))
        T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    ])

    # Load The Data
    # Annotation File directories 
    training_file = "./LISA_DATA/Training/training_annotations.csv"
    validation_file = "./LISA_DATA/Validating/validating_annotations.csv"
    testing_file = "./LISA_DATA/Testing/testing_annotations.csv" 

    # Traffic Sign Image Directories 
    training_images = "./LISA_DATA/Training/training_images"
    validating_images = "./LISA_DATA/Validating/validating_images"
    testing_images = "./LISA_DATA/Testing/testing_images"

    # Initializing the custom image dataset
    training_data = CustomImageDataset(training_file, training_images, data_transforms)
    validation_data = CustomImageDataset(validation_file, validating_images, data_transforms)
    testing_data = CustomImageDataset(testing_file, testing_images, data_transforms)

    # ---------------------------------------------------------------------------------------
    # VISUALIZING DATASETS
    # ---------------------------------------------------------------------------------------

    # Training set distribution visualization 
    # Making a list to count the occurrence for each class 
    occurrence_list = [0] * num_classes
    
    with open(training_file, 'r') as read_obj:
        csv_reader = DictReader(read_obj)
        for row in csv_reader:
            class_value = int(row['ClassID'])
            count = occurrence_list[class_value]
            count += 1
            occurrence_list[class_value] = count

    # For the 47 classes
    a = np.arange(47)
    fig, ax = plt.subplots(figsize=(12,10), edgecolor='k')
    ax.set_xticks(a)
    # Getting the labels from the 'labels_map' dictionary
    values = list(labels_map.values())
    ax.set_xticklabels(values, rotation=90)
    ax.set_title('Training Set Class Distribution')

    plt.xlabel('Class')
    plt.ylabel('Count')
    ax.bar(values, occurrence_list)
    plt.show()

    # Testing set distribution visualization 
    testing_occurrence_list = [0] * num_classes

    with open(testing_file, 'r') as read_obj:
        csv_reader = DictReader(read_obj)
        for row in csv_reader:
            class_value = int(row['ClassID'])
            count = testing_occurrence_list[class_value]
            count += 1
            testing_occurrence_list[class_value] = count

    a = np.arange(47)
    fig, ax = plt.subplots(figsize=(12,10), edgecolor='k')
    ax.set_xticks(a)
    # Getting the labels from the 'labels_map' dictionary
    values = list(labels_map.values())
    ax.set_xticklabels(values, rotation=90)
    ax.set_title('Testing Set Class Distribution')

    plt.xlabel('Class')
    plt.ylabel('Count')
    ax.bar(values, testing_occurrence_list)
    plt.show()

    return training_data, validation_data, testing_data

class EdLeNet(nn.Module):
    def __init__(self, l1=120, l2=47):
        super(EdLeNet, self).__init__()

        # Describing the 3 convolutional layers
        self.layer1 = nn.Sequential(
            # Padding = valid is the same as no padding 
            # 1st param: # of input channels (grayscale, so 1) (change when using color images)
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
        self.fc1 = nn.Linear(2 * 2 * 128, l1)
        # 3 is the number of classes that we are using at the moment
        self.fc2 = nn.Linear(l1, l2)

    # Define how the data flows through these layers when performing a forward pass
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # Reshaping 2x2x128 into 512x1
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out 

def train_EdLeNet(config, checkpoint_dir=None, data_dir=None):
    net = EdLeNet(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(EdLeNet)
    net.to(device)
    print(device)

    criterion = nn.CrossEntropyLoss()
    # TODO: Add momentum to the optimizer? What will it do (example: momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    # Checkpointing is optional 
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    training_data, validation_data, testing_data = load_data()

    # TODO: Check what num workers means
    # num_workers = 8
    train_loader = DataLoader(training_data, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)
    valid_loader = DataLoader(validation_data, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8)

    for epoch in range(10):
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data 
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            # Zero the parameter gradients 
            optimizer.zero_grad()

            # forward + backward + optimize 
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999: # Print every 2000 mini-batches 
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, 
                                                running_loss / epoch_steps))
                running_loss = 0.0
            
        val_loss = 0.0
        val_steps = 0 
        total = 0
        correct = 0 
        for i, data in enumerate(valid_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        
        tune.report(loss=(val_loss / val_steps), accuracy = correct / total)
    print("Finished Training")

def test_accuracy(net, device="cpu"):
    training_data, validation_data, testing_data = load_data()

    test_loader = DataLoader(testing_data, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data 
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    load_data()

    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size":tune.choice([2, 4, 8, 16])
    }
    
    scheduler = ASHAScheduler(
        metric = "loss",
        mode = "min",
        max_t = max_num_epochs,
        grace_period = 1,
        reduction_factor = 2
    )
    reporter = CLIReporter(
        # parameter_columns = ["l1", "l2", "lr", "batch_size"],
        # CHECK if we are supposed to be printing losses
        metric_columns = ["loss", "accuracy", "training_iteration"]
    )
    result = tune.run(
        partial(train_EdLeNet),
        resources_per_trial = {"cpu": 2, "gpu": gpus_per_trial},
        config = config,
        num_samples = num_samples,
        scheduler = scheduler,
        progress_reporter = reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = EdLeNet(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)

'''
loss_values = []
accuracy_values = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # Calling the training and testing functions on each epoch
    training_loss = train(train_loader, model, criterion, optimizer, loss_values)
    testing_accuracy = test(test_loader, model, criterion, accuracy_values)
# Print done once all the epochs have been iterated through 
print("Done!")

# Now to print some analytic graphs 
# TODO: Loss curve & accuracy curve (possibly add uncertainty as well)
# Loss:
plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(training_loss, label="train")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy:
plt.figure(figsize=(10, 5))
plt.title("Test Accuracy")
plt.plot(testing_accuracy, label="test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
'''