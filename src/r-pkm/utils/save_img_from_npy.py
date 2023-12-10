import numpy as np
import os
from PIL import Image

# please note that those depth images are int32 (0, 65535), the controlnet input are int8 (0, 255)

#The same for all subjects
images = np.load('data/processed_depth_data/subj01/nsd_test_stim_sub1.npy')

test_images_dir = 'data/nsddata_stimuli/test_depth_tmp/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    # im = Image.fromarray(images[i])#.astype(np.uint8)
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))