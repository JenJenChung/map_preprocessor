#!/usr/bin/env python

import models
import utils
import random
import torch
import numpy as np
from torch.autograd import Variable
import torchvision

# Model to test
model_name = 'map_ae_256x256_relu_20_MSE'

# Load the model parameter
params = np.load('models/trained_models/' + model_name + '_params.npy')
params = params.item()

# Load the model and its learnt parameters
autoencoder = models.ModelAErelu(params['code_size'], params['image_width'], params['image_height'])
autoencoder.load_state_dict(torch.load('models/trained_models/' + model_name + '.model', map_location=lambda storage, loc: storage))

# Check if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('INFO: Start training on device %s' % device)
autoencoder.to(device)

# Dataset files
trainset_name = 'data/converted_maps_train_filtered.tar'
validationset_name = 'data/converted_maps_val_filtered.tar'
testset_name = 'data/converted_maps_test_filtered.tar'

# Load datasets for training, validation and testing
num_workers = 4
train_set = utils.MapDataset(trainset_name)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=num_workers)
validation_set = utils.MapDataset(validationset_name)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=num_workers)
test_set = utils.MapDataset(testset_name)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)

# define loss function
loss_fn_L1 = torch.nn.L1Loss()
loss_fn_MSE = torch.nn.MSELoss()

with torch.no_grad():
    loss_L1_train = 0.0
    loss_MSE_train = 0.0
    for data in train_loader:
        data = data.to(device)
        outputs, _ = autoencoder(data)
        loss_L1_train += loss_fn_L1(outputs, data)
        loss_MSE_train += loss_fn_MSE(outputs, data)

    loss_L1_val = 0.0
    loss_MSE_val = 0.0
    for data in validation_loader:
        data = data.to(device)
        outputs, _ = autoencoder(data)
        loss_L1_val += loss_fn_L1(outputs, data)
        loss_MSE_val += loss_fn_MSE(outputs, data)

    loss_L1_test = 0.0
    loss_MSE_test = 0.0
    for data in test_loader:
        data = data.to(device)
        outputs, _ = autoencoder(data)
        loss_L1_test += loss_fn_L1(outputs, data)
        loss_MSE_test += loss_fn_MSE(outputs, data)

    # Evaluate the model performance on the test set
    for i in range(50):
        # Try reconstructing on test data
        test_image = random.choice(test_set)
        test_image = torch.tensor(test_image)
        test_image = Variable(test_image[0].view([1, 1, params['image_width'], params['image_height']]))
        test_image = test_image.to(device)
        test_reconst, _ = autoencoder(test_image)

        torchvision.utils.save_image(test_image.data, (str(i) + '_orig.png'))
        torchvision.utils.save_image(test_reconst.data, (str(i) + '_reconst.png'))

print('INFO: Average L1 loss on training set: %s' % (loss_L1_train.item()/len(train_set)))
print('INFO: Average MSE loss on training set: %s' % (loss_MSE_train.item()/len(train_set)))
print('INFO: Average L1 loss on validation set: %s' % (loss_L1_val.item()/len(validation_set)))
print('INFO: Average MSE loss on training set: %s' % (loss_MSE_val.item()/len(validation_set)))
print('INFO: Average L1 loss on test set: %s' % (loss_L1_test.item()/len(test_set)))
print('INFO: Average MSE loss on test set: %s' % (loss_MSE_test.item()/len(test_set)))
