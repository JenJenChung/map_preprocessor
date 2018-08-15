#!/usr/bin/env python

import models
import utils
import random
import torch
import numpy as np
import skimage.measure
import copy
from torch.autograd import Variable
import torchvision

import rospy
from map_preprocessor.msg import LocalMap
from map_preprocessor.msg import MapEncoding


class MapEncode:
  def __init__(self):
    print('Creating encoder object...')
    rospy.init_node('map_encoder')
    
    # Read in maxpool size
    self.maxpool = rospy.get_param('maxpool')
    
    # Read in model name and directory from rosparam
    self.model_name = rospy.get_param('model_name')
    self.directory_name = rospy.get_param('directory_name')
    
    # Load the model parameters
    self.params = np.load(self.directory_name + self.model_name + '_params.npy')
    self.params = self.params.item()
    
    if 'selu' in self.model_name:
      print('Autoencoder uses selu activation')
      self.autoencoder = models.ModelAEselu(self.params['code_size'], self.params['image_width'], self.params['image_height'])
    elif 'relu' in self.model_name:
      print('Autoencoder uses relu activation')
      self.autoencoder = models.ModelAErelu(self.params['code_size'], self.params['image_width'], self.params['image_height'])
    else:
      print('Model %s not found' % self.model_name)
    
    self.autoencoder.load_state_dict(torch.load(self.directory_name + self.model_name + '.model', map_location=lambda storage, loc: storage))
    
    # Check if GPU is available
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('INFO: Using device %s' % self.device)
    self.autoencoder.to(self.device)
    
    # Subscribers and publishers
    self.map_sub_ = rospy.Subscriber('local_map', LocalMap, self.map_callback)
    self.code_pub_ = rospy.Publisher('map_encoding', MapEncoding, queue_size = 10)
    
    
  def map_callback(self, msg):
    # Get map data, resize and convert to tensor type
    local_map = copy.deepcopy(msg.data)
    local_map = np.asarray(local_map, dtype="float32") ;
    test_map = local_map.reshape(msg.info.height, msg.info.width)
    test_map = skimage.measure.block_reduce(test_map, (self.maxpool, self.maxpool), np.max)
    test_map = torch.tensor(test_map)
    test_map = Variable(test_map.view([1, 1, self.params['image_width'], self.params['image_height']]))
    test_map = test_map.to(self.device)
    
    with torch.no_grad():
      # Compute encoding
      code = self.autoencoder.encode(test_map)
    
    # Publish encoding
    encoding = MapEncoding()
    encoding.header.seq = msg.header.seq
    encoding.header.stamp = msg.header.stamp
    encoding.header.frame_id = msg.header.frame_id
    np_code = np.array(code, dtype="float32")
    encoding.data = np_code[0]
#    print(encoding)
    self.code_pub_.publish(encoding)


if __name__ == '__main__':
  map_encode = MapEncode()
  rospy.spin()

