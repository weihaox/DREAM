import numpy as np
import os
from PIL import Image

#The same for all subjects
images = np.load('data/processed_data/subj01/nsd_train_stim_sub1.npy')

test_images_dir = 'data/nsddata_stimuli/train_images/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))