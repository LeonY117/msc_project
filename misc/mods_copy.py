import os
from PIL import Image
from tqdm import tqdm

# Resize all images in MODS_DIR to 512x384

MODS_DIR = '../dataset/mods/sequences'
MODS_DIR_RESIZED = '../dataset/mods_small/sequences'

for sequence in tqdm(os.listdir(MODS_DIR)):
    # copy the entire imus folder
    sequence_dir = os.path.join(MODS_DIR, sequence, 'imus')
    sequence_dir_resized = os.path.join(MODS_DIR_RESIZED, sequence, 'imus')
    # Create resized sequence directory
    if not os.path.exists(sequence_dir_resized):
        os.makedirs(sequence_dir_resized)
    # Resize all images in sequence
    for image in os.listdir(sequence_dir):
        # Get image paths
        image_path = os.path.join(sequence_dir, image)
        image_path_resized = os.path.join(sequence_dir_resized, image)
        # Resize image
        with Image.open(image_path) as img:
            img.save(image_path_resized)
    
    if 'ignore_mask.png' in os.listdir(os.path.join(MODS_DIR, sequence)):
        # copy the image to the resized folder
        image_path = os.path.join(MODS_DIR, sequence, 'ignore_mask.png')
        image_path_resized = os.path.join(MODS_DIR_RESIZED, sequence, 'ignore_mask.png')
        with Image.open(image_path) as img:
            img.save(image_path_resized)


