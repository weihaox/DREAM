import numpy as np
import os
from PIL import Image
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

images_dir = 'train_images_processed_midas'
output_npy_file = 'data/processed_depth_data/subj01/nsd_train_stim_sub1.npy'

images = []
for filename in sorted_alphanumeric(os.listdir(images_dir)):
    if filename.endswith(".png"):
        img_path = os.path.join(images_dir, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        img_array = np.uint8(img_array)

        # the following two lines are for validation
        # im = Image.fromarray(img_array)
        # im.save('{}/{}'.format('data/nsddata_stimuli/test_depth',filename))

        images.append(img_array)

images = np.array(images)

# Save the numpy array to a file
np.save(output_npy_file, images)
