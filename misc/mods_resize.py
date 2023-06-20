import os
from PIL import Image
from tqdm import tqdm

# Resize all images in MODS_DIR to 512x384

MODS_DIR = '../dataset/mods/sequences'
MODS_DIR_RESIZED = '../dataset/mods_small/sequences'

for sequence in tqdm(os.listdir(MODS_DIR)):
    print('Resizing sequence: ' + sequence)
    sequence_dir = os.path.join(MODS_DIR, sequence, 'frames')
    sequence_dir_resized = os.path.join(MODS_DIR_RESIZED, sequence, 'frames')
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
            resized_image = img.resize((512, 384))
            resized_image.save(image_path_resized)

