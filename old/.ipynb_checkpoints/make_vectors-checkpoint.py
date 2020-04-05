import tensorflow as tf
from tqdm import tqdm
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import pandas as pd
import string



# Download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    os.remove(annotation_zip)

# Download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
      PATH = os.path.abspath('.') + image_folder
        
        
        
# Read the json file

annotation_file ='./annotations/captions_train2014.json'
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'].translate(str.maketrans('', '', string.punctuation)) + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)


print(len(all_img_name_vector))
print(len(all_captions))


### ADD EXTRA DISNEY
### CHECK FOR NOT JPEG FILES

not_jpg = []

disney_words = []

disney=pd.read_csv('how_to.csv').to_dict(orient='records')
files = frozenset(os.listdir('./disney_img/'))
for item in tqdm(disney):
    file_name = item['im_id']
    if ('jpg' in file_name) and (file_name in files):
        try:
            caption =  caption = '<start> ' + item['title'] + ' <end>'
            file_path = './disney_img/'+item['im_id']
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            all_img_name_vector.append(file_path)
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            all_captions.append(caption)
            disney_words+=[w.lower() for w in caption.split(' ')]
        except:
            not_jpg.append(file_path)
            
            
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)



def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path
    

print('SAVING VECTORS')

# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)

imge_vectors_paths = []

for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                                  (batch_features.shape[0], -1, batch_features.shape[3]))

    
    for bf, p in zip(batch_features, path):
        feature_name = p.numpy().decode("utf-8").split('/')[-1]
        feature_path = './features/'+feature_name
        np.save(feature_path, bf.numpy())
        imge_vectors_paths.append(feature_path)