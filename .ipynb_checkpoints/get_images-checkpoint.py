from tqdm import tqdm
import requests
from datetime import datetime
import tensorflow as tf
import pandas as pd
import time
import os
from PIL import Image


IMG_FOLDER = './disney_img/'


data = pd.read_csv('how_to_links.csv')

print('all records: ',data.shape[0])
def save_img(url, name, folder='images'):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(folder+'/'+name, 'wb') as f:
            for chunk in r:
                f.write(chunk)
        return True
    else:
        return False
        
        
images_saved = []
for i, image in enumerate(data.to_dict(orient='records')):
    name = image['im_id']
    if os.path.isfile(os.path.join(folder, name)):
        images_saved.append(os.path.join(folder, name.replace('png','jpg')))
        continue
    
    ## crop images and save as jpg
    try:
        if save_img(image['link'],name, folder=IMG_FOLDER):
            im = Image.open(os.path.join(IMG_FOLDER, name))
            w, h = im.size
            im = im.crop((0,0,w,h-50))
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.join(IMG_FOLDER, name.replace('png','jpg')))
            if 'png' in name:
                os.remove(os.path.join(IMG_FOLDER, name))
            images_saved.append(os.path.join(IMG_FOLDER, name))
    except Exception as e:
        print(image['link'], e)
        
    if i%300 ==0:
        print(f'current num {i}, images added {len(images_saved)}')

        

## PREPARE CSV WITH READABLE FILES

max_len = 14
min_len = 4

disney=pd.read_csv('how_to_new.csv')
disney['ext'] = disney['im_id'].apply(lambda r: r.split('.')[-1])
disney = disney[disney['ext']=='jpg']


not_jpg =[]
disney_words = []
disney_captions = []
disney_images = []

records=disney.to_dict(orient='records')
files = frozenset(os.listdir(IMG_FOLDER))
for item in tqdm(records):
    file_name = item['im_id']
    if file_name in files:
        
        # try to open jpg files and add only readable ones
        
        try:
            caption = ' '.join((item['title'].translate(str.maketrans('', '', string.punctuation))).split(' '))
            caption  = '<start> ' + caption.lower() + ' <end>'
            n = len(caption.split(' ')) 
            if (min_len > n) or (n > max_len ):
                continue
            file_path = './disney_img/'+item['im_id']
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
#             all_img_name_vector.append(file_path)
#             all_captions.append(caption)
            disney_words+=[w.lower() for w in caption.split(' ')]
            disney_captions.append(caption)
            disney_images.append(file_path)
        except:
            not_jpg.append(file_path)
            
            
pd.DataFrame(zip(disney_captions, disney_images), columns=['caption','path']).to_csv('clean_cpations_n_files.csv')