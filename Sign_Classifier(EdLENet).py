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
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import torch 
from torch import nn
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.layers import flatten
from sklearn.utils import shuffle


models_path = "./models/"

# NOTE: img_dir was removed from init and therefore getitem since 
# the image path is included inside of the .csv annotation file 
# Creating a custom Dataset for your files 
class CustomImageDataset(Dataset):
    # Run when instantiating the Dataset object 
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # transform and target_transform specify the feature and label transform actions
        self.img_labels = pd.read_csv(annotations_file) # Reads annotations from csv file into labels
        self.img_dir = img_dir
        self.transform = transform 
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
        img_path = os.path.join(self.img_dir, picture_path)
        # Takes image's location on disk, converts that to a tensor using read_image
        image = read_image(img_path)
        # Retrieves the corresponding label from the csv data 
        label = self.img_labels.iloc[idx, 1]
        # Calls the transform functions if applicable 
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #print(img_path)
        # Returns the tensor image and corresponding label in a tuple
        return image, label 



# Utility class that stores important config info about our model
class ModelConfig:
    def __init__(self, model, name, input_img_dimensions, 
    conv_layers_config, fc_output_dims, output_classes, dropout_keep_pct):
        self.model = model
        self.name = name
        self.input_img_dimensions = input_img_dimensions

        self.conv_filter_size = conv_layers_config[0] # wxh dimension of filters
        self.conv_depth_start = conv_layers_config[1] # starting depth (doubles each layer) 
        self.conv_layers_count = conv_layers_config[2] # Number of convolutional layers

        self.fc_output_dims = fc_output_dims
        self.output_classes = output_classes

        # Try with different values for drop out at convolutional and fully connected layers
        self.dropout_conv_keep_pct = dropout_keep_pct[0]
        self.dropout_fc_keep_pct = dropout_keep_pct[1]

class ModelExecutor:
    def __init__(self, model_config, learning_rate=0.001):
        self.model_config = model_config
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with g.name_scope(self.model_config.name) as scope:
                # Create Model operations 
                self.create_model_operations()

                # Create a saver to persist the results of execution 
                self.saver = tf.train.Saver()

    # Defining our placeholder variables
    def create_placeholders(self):
        input_dims = self.model_config.input_img_dimensions
        self.x = tf.placeholder(tf.float32, (None, input_dims[0], input_dims[1], input_dims[2]),
        name="{0}_x".format(self.model_config.name))
        self.y = tf.placeholder(tf.int32, (None), name="{0}_y".format(self.model_config.name))
        self.one_hot_y = tf.one_hot(self.y, self.model_config.output_classes)

        self.dropout_placeholder_conv = tf.placeholder(tf.float32)
        self.dropout_placeholder_fc = tf.placeholder(tf.float32)

    # Sets up all operations needed to execute deep learning pipeline
    def create_model_operations(self):
        # First step is to set our x, y, etc 
        self.create_placeholders()
        cnn = self.model_config.model

        # Build the network
        self.logits = cnn(self.x, self.model_config, self.dropout_placeholder_conv, self.dropout_placeholder_fc)
        # Use softmax as the activation as the function for final layer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        # Combined all the losses across batches
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        # What method do we use to reduce our loss?
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # Attempt to reduce the loss using our chosen optimizer  
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Get the top prediction for model against labels and check whether they match 
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))

        # Compute accuracy at batch level 
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Compute what the prediction would be, when we don't have matching label
        self.prediction = tf.argmax(self.logits, 1)
        # Registering our top 5 predictions 
        self.top5_predictions = tf.nn.top_k(tf.nn.softmax(self.logits), k=3, sorted=True, name=None)

    # Evaluates the model's accuracy and loss for the supplied dataset
    # Dropout is ignored in this case (dropout_keep_pct to 1.0)
    def evaluate_model(self, X_data, Y_data, batch_size):
        num_examples = len(X_data)
        total_accuracy = 0.0
        total_loss = 0.0 
        sess = tf.get_default_session()

        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], Y_data[offset:offset+batch_size]
            
            # Compute both accuracy and loss for this batch 
            accuracy = sess.run(self.accuracy_operation,
                                feed_dict={
                                    self.dropout_placeholder_conv: 1.0,
                                    self.dropout_placeholder_fc: 1.0,
                                    self.x: batch_x,
                                    self.y: batch_y
                                })
            loss = sess.run(self.loss_operation, 
                            feed_dict={
                                self.dropout_placeholder_conv: 1.0,
                                self.dropout_placeholder_fc: 1.0,
                                self.x: batch_x,
                                self.y: batch_y
                            })
            
            # Weighting accuracy by the total number of elements in batch 
            total_accuracy += (accuracy * len(batch_x))
            total_loss += (loss * len(batch_x))
        
        # To produce a true mean accuracy over whole dataset 
        return (total_accuracy / num_examples, total_loss / num_examples)

    # NOTE: The default batch size as well as the epochs!
    def train_model(self, X_train_features, X_train_labels, X_valid_features, y_valid_labels, 
                    batch_size=512, epochs=100, PRINT_FREQ=10):

        # Create our array of metrics 
        training_metrics = np.zeros((epochs, 3))
        validation_metrics = np.zeros((epochs, 3))

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train_features)

            print("Training {0} [epochs={1}, batch_size={2}]...\n".format(self.model_config.name,
                epochs, batch_size))
            
            for i in range(epochs):
                start = time.time()
                X_train, Y_train = shuffle(X_train_features, X_train_labels)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={
                        self.x: batch_x,
                        self.y: batch_y,
                        self.dropout_placeholder_conv: self.model_config.dropout_conv_keep_pct,
                        self.dropout_placeholder_fc: self.model_config.dropout_fc_keep_pct,

                    })
                
                end_training_time = time.time()
                training_duration = end_training_time - start

                # Computing training accuracy
                training_accuracy, training_loss = self.evaluate_model(X_train_features, X_train_labels, batch_size)

                # Computing validation accuracy 
                validation_accuracy, validation_loss = self.evaluate_model(X_valid_features, y_valid_labels, batch_size)

                end_epoch_time = time.time()
                validation_duration = end_epoch_time - end_training_time
                epoch_duration = end_epoch_time - start

                if i == 0 or (i+1) % PRINT_FREQ == 0:
                    print("[{0}]\ttotal={1:.3f}s | train: time={2:.3f}s, loss={3:.4f}, acc={4:.4f} | val: time={5:.3f}s, loss={6:.4f}, acc={7:.4f}"
                    .format(i+1, epoch_duration, training_duration, training_loss, training_accuracy, 
                    validation_duration, validation_loss, validation_accuracy))

                training_metrics[i] = [training_duration, training_loss, training_accuracy]
                validation_metrics[i] = [validation_duration, validation_loss, validation_accuracy]

            model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
            # Save the model
            self.saver.restore(sess, model_file_name)
            print("Model {0} saved".format(model_file_name))

        return (training_metrics, validation_metrics, epoch_duration)

    # NOTE: Default batch size 
    # Evaluates the model with the test dataset and test labels
    # Returns the tuple (test_accuracy, test_loss, duration)
    def test_model(self, test_imgs, test_lbs, batch_size=512):
            with tf.Session(graph = self.graph) as sess:
                # Never forget to re-initialise the variables 
                tf.global_variables_initializer()

                model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
                self.saver.restore(sess, model_file_name)

                start = time.time()
                (test_accuracy, test_loss) = self.evaluate_model(test_imgs, test_lbs, batch_size)
                duration = time.time() - start 
                print("[{0} - Test Set]\ttime={1:.3f}s, loss={2:.4f}, acc={3:.4f}"
                .format(self.model_config.name, duration, test_loss, test_accuracy))

            return (test_accuracy, test_loss, duration)

    # Returns the predictions associated with a bunch of images 
    def predict(self, imgs, top_5=False):
        preds = None
        with tf.Session(graph = self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()

            model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
            self.saver.restore(sess, model_file_name)

            if top_5:
                preds = sess.run(self.top5_predictions, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                    })
            else:
                preds = sess.run(self.prediction, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                    })

        return preds

    # Shows the resulting feature maps at a given convolutional level for a single image
    def show_conv_feature_maps(self, img, conv_layer_idx=0, activation_min=-1, activation_max=-1,
                                plt_num=1, fig_size=(15, 15), title_y_pos=1.0):

        #s = tf.train.Saver()
        with tf.Session(graph = self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()
            #tf.reset_default_graph()

            model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
            self.saver.restore(sess, model_file_name)

            # Run a prediction
            preds = sess.run(self.prediction, feed_dict={
                self.x: np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]]),
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0
                })

            var_name = "{0}/conv_{1}_relu:0".format(self.model_config.name, conv_layer_idx)
            print("Fetching tensor: {0}".format(var_name))
            conv_layer = tf.get_default_graph().get_tensor_by_name(var_name)

            activation = sess.run(conv_layer, feed_dict={
                self.x: np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]]),
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0
                })
            featuremaps = activation.shape[-1]
            # (1, 13, 13, 64)
            print("Shape of activation layer: {0}".format(activation.shape))

            # fix the number of columns 
            cols = 8 
            rows = featuremaps // cols 
            fig, axes = plt.subplots(rows, cols, figsize=fig_size)
            k = 0
            for i in range(0, rows):
                for j in range(0, cols):
                    ax = axes[i, j]
                    featuremap = k

                    if activation_min != -1 & activation_max != -1:
                        ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, vmax=activation_max, cmap="gray")
                    elif activation_max != -1:
                        ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
                    elif activation_min != -1:
                        ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
                    else:
                        ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
                        ax.axis("off")
                        k += 1

            fig.suptitle("Feature Maps at layer: {0}".format(conv_layer), fontsize=12, fontweight='bold', y=title_y_pos)
            fig.tight_layout()
            plt.show()

# EdleNet implementation
# The second parameter, which encapsulates model configuration, enables varying the convolution filter sizes
# As well as well as the number of fully connected layers and their output dimensions 
# The 3rd and 4th parameters represent dropout placeholders for convolutional and fully connected layers respectively
def EdLeNet(x, mc, dropput_conv_pct, dropout_fc_pct):
    # Used for randomly defining weights and biases 
    mu = 0
    sigma = 0.1 

    prev_conv_layer = x
    conv_depth = mc.conv_depth_start
    conv_input_depth = mc.input_img_dimensions[-1]

    print("[EdLeNet] Building neural network [conv layers={0}, conv filter size={1}, conv start depth={2}, fc layers={3}]"
    .format(mc.conv_layers_count, mc.conv_filter_size, conv_depth, len(mc.fc_output_dims)))

    for i in range(0, mc.conv_layers_count):
        # Layer depth grows exponentially 
        conv_output_depth = conv_depth * (2 ** (i))
        conv_W = tf.Variable(tf.truncated_normal(shape=(mc.conv_filter_size, mc.conv_filter_size, conv_input_depth, conv_output_depth), 
        mean = mu, stddev = sigma))
        conv_b = tf.Variable(tf.zeros(conv_output_depth))

        conv_output = tf.nn.conv2d(prev_conv_layer, conv_W, strides=[1, 1, 1, 1], padding='VALID', name="conv_{0}"
                        .format(i)) + conv_b
        conv_output = tf.nn.relu(conv_output, name="conv_{0}_relu".format(i))
        # Traditional max 2x2 pool
        conv_output = tf.nn.max_pool(conv_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
       
        # Apply dropout - even at the conv level 
        conv_output = tf.nn.dropout(conv_output, dropput_conv_pct)

        # Setting our loop variables accordingly 
        prev_conv_layer = conv_output
        conv_input_depth = conv_output_depth

    # Flatten results of second convolutional layer so that it can be supplied to fully connected layer
    fc0 = flatten(prev_conv_layer)

    # Now creating our fully connected layers
    prev_layer = fc0
    for output_dim in mc.fc_output_dims:
        fcn_W  = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], output_dim), 
                                                 mean = mu, stddev = sigma))
       
        fcn_b = tf.Variable(tf.zeros(output_dim))

        prev_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(prev_layer, fcn_W) + fcn_b), dropout_fc_pct)

    # Final layer (Fully Connected)
    fc_final_W  = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], mc.output_classes), 
                                                  mean = mu, stddev = sigma))
    fc_final_b  = tf.Variable(tf.zeros(mc.output_classes))
    logits = tf.matmul(prev_layer, fc_final_W) + fc_final_b
    
    return logits

def main():
    # Performs preprocessing on the dataset images 
    data_transforms = T.Compose([
        T.Resize((32,32)),
        T.Grayscale(1)
    ])

    # STEP 0: Load The Data
    # annotation files 
    training_file = "./LISA_DATA/Training/training_annotations.csv"
    validation_file = "./LISA_DATA/Validating/validating_annotations.csv"
    testing_file = "./LISA_DATA/Testing/testing_annotations.csv" 

    # image directories 
    training_images = "./LISA_DATA/Training/training_images"
    validating_images = "./LISA_DATA/Validating/validating_images"
    testing_images = "./LISA_DATA/Testing/testing_images"

    training_data = CustomImageDataset(training_file, training_images,data_transforms)
    validation_data = CustomImageDataset(validation_file, validating_images, data_transforms)
    testing_data = CustomImageDataset(testing_file, testing_images, data_transforms)

    # Using __getitem__ in a loop of our dataset(annotations.csv length)
    print(training_data.__getitem__(0))
    print("\n")

    # Each element in the dataloader iterable 
    # will return a batch of 64 features and labels 
    batch_size = 64

    # Loading data using dataloaders
    # Reshuffle the data at every epoch to reduce model overfitting 
    # If using CUDA, num_workers should be set to 1 and pin_memory to True  
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

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
    for (X, y) in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        #print("\nShape of y: ", y.shape, y.dtype)
        break

    # Exploratory visualization of the dataset 
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        #plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

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

    valid_features, valid_labels = next(iter(valid_dataloader))
    test_features, test_labels = next(iter(test_dataloader))

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
    '''

    # Step 2: Design and Test a Model Architecture 
    # Get Device for Training 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    mc_3x3 = ModelConfig(EdLeNet, "EdLeNet_3x3_Color_Sample", [32, 32, 1], [3, 16, 3], [120, 84], 3, [1.0, 1.0])
    me_c_sample_3x3 = ModelExecutor(mc_3x3)

    # Passing X and Y features over to the model
    (c_sample_3x3_tr_metrics, c_sample_3x3_val_metrics, c_sample_3x3_duration) = me_c_sample_3x3.train_model(train_features, train_labels, valid_features, valid_labels, epochs=50)
    (c_sample_3x3_ts_accuracy, c_sample_3x3_ts_loss, c_sample_3x3_ts_duration) = me_c_sample_3x3.test_model(test_features, test_labels)

main()