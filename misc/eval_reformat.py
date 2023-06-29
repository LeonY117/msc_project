# Reformat the network outputs to the format required by the evaluation script

import os
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm

def main():
    # accept arguments to the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)

    args = parser.parse_args()

    # get the input directory
    input_dir = os.path.join('../results/', args.input_dir)
    out_dir = os.path.join('../results/', 'eval_' + args.input_dir)
    
    # read the images and resize them to 512x384, and increase channel to 3
    for sequence in tqdm(os.listdir(input_dir)):
        sequence_dir = os.path.join(input_dir, sequence)
        for image in os.listdir(sequence_dir):
            image_path = os.path.join(sequence_dir, image)
            with Image.open(image_path) as img:
                # resize to 512x384
                resized_image = img.resize((512, 384))

                # map pixel values 0 -> (0, 0, 0), 1 -> (255, 0, 0), 2 -> (0, 255, 0)
                # this is the format required by the evaluation script
                # code to do the above:
                resized_image = resized_image.convert('RGB')
                
                # convert resized_image into numpy array:
                image_arr = np.array(resized_image).astype(np.uint8)
                for i in range(image_arr.shape[0]):
                    for j in range(image_arr.shape[1]):
                        if image_arr[i, j, 0] == 0:
                            image_arr[i, j] = [0, 0, 0]
                        elif image_arr[i, j, 0] == 1:
                            image_arr[i, j] = [255, 0, 0]
                        elif image_arr[i, j, 0] == 2:
                            image_arr[i, j] = [0, 255, 0]
                
                # save the image       
                out_sequence_dir = os.path.join(out_dir, sequence)
                if not os.path.exists(out_sequence_dir):
                    os.makedirs(out_sequence_dir)
                out_image_path = os.path.join(out_sequence_dir, image)

                # convert image_arr back to PIL image
                resized_image = Image.fromarray(image_arr)

                resized_image.save(out_image_path)

# usage:
# python eval_reformat.py --input_dir test_results

if __name__ == '__main__':
    main()