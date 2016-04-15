# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:42:49 2016

@author: sandouku
"""
'''
from IPython import display
def listFilesInDir(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath= 'C:\\Users\\sandouku\\TensorFlow\\tensorflow\\tensorflow\\examples\\udacity\\notMNIST_large\\A'
As = listFilesInDir(mypath)

for i in range(10):
    imagePath = join(mypath, As[i])
    display.display(display.Image(imagePath))
'''

import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from os import listdir
from os.path import isfile, join


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, max_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  image_files = image_files[:min(len(image_files),max_num_images)];
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  for image_index, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index + 1
  dataset = dataset[0:num_images, :, :]
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset



if __name__ == "__main__":
    '''
    mypath= 'C:\\Users\\sandouku\\TensorFlow\\tensorflow\\tensorflow\\examples\\udacity\\notMNIST_large\\A'
    dataset = load_letter(mypath, 10)
    

    for i in range(dataset.shape[0]):
        plt.figure()
        plt.imshow(dataset[i,:,:])
    '''
    '''
    mypath= 'C:\\Users\\sandouku\\TensorFlow\\tensorflow\\tensorflow\\examples\\udacity\\notMNIST_large\\'
    pickle_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    sizes_array = list()
    for i in pickle_files:
        if not i.endswith('.pickle'): continue
        f  = open(mypath + i, 'rb')
        dataset = pickle.load(f)
        sizes_array.append(dataset.shape[0])
        f.close()
    # find maximum deviation from the mean
    diff_array = np.abs(np.subtract(sizes_array, np.median(sizes_array)))
    print "maximum divuiation from the median num of examples is: ", max(diff_array)
    '''
    mypath= 'C:\\Users\\sandouku\\TensorFlow\\tensorflow\\tensorflow\\examples\\udacity\\'
    with open((mypath+'notMNIST.pickle'),'rb') as f:
        dataset = pickle.load(f)
    def Compare2DImages(I1, I2):
        for i in range(I1.shape[0]):
            for j in range(I1.shape[1]):
                if (I1[i,j] != I2[i,j]): return False
        return True
        #return sum(sum((I1 == I2)==False)) == 0
    def FindDuplicateImages(set1, set2):
        set2_in_set1_filter = np.ones((set2.shape[0],1))
        for i in range(set2.shape[0]):
            for j in range(set1.shape[0]):
                if (Compare2DImages(set2[i,:,:], set1[j,:,:])):
                    set2_in_set1_filter[i] = False
                    break
        return set2_in_set1_filter
    
    valid_in_training_filter = FindDuplicateImages(dataset['train_dataset'], dataset['valid_dataset'])
    test_in_training_filter = FindDuplicateImages(dataset['train_dataset'], dataset['test_dataset'])
    
    valid_dataset = dataset['valid_dataset'][valid_in_training_filter,:,:]
    valid_labels = dataset['valid_labels'][valid_in_training_filter,:,:]
    