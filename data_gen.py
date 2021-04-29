import multiprocessing as mp
import threading 
import numpy as np
import random
import argparse
from tqdm import tqdm
from generate_one_image import generate_one_training_image, DownsamplingImage
from PIL import Image
import cv2
import json
import os


parser = argparse.ArgumentParser(description='Generate synthetic training images.')
parser.add_argument('--num_processes', '-n', type=int, default=6,
                    help='Number of processes used for image generation')
parser.add_argument('--train_size', '-t', type=int, default=3000,
                    help='Number of training images to be generated')

parser.add_argument('--start_index', '-s', type=int, default=0,
                    help='training image starting index')
args = parser.parse_args()

max_num_barcodes_per_image = 10
barcode_border = 5
min_im_size = 400
max_im_size = 400

output_dir = "low_resolution"
generated_type = "train"

def generate(generated, train_size, start_index):
    np.random.seed(start_index + generated)
    random.seed(start_index + generated)
    ind = generated / (train_size - generated)
    bbxes = {}
    pbar = tqdm(total=train_size-generated)
    while generated < train_size:
        # if os.path.exists("{}/roi_{}_im/roi{}.png".format(output_dir, generated_type, generated+start_index)):
        #     num_barcodes = np.random.randint(1, max_num_barcodes_per_image+1)
        #     final_size = np.random.randint(min_im_size, max_im_size+1)
        #     generated += 1
        #     pbar.update(1)
        #     continue
        try:
            num_barcodes = np.random.randint(1, max_num_barcodes_per_image+1)
            final_size = np.random.randint(min_im_size, max_im_size+1)
            train, train_mask, segmented_single_masks, all_bbx = generate_one_training_image(
                                                                    num_barcodes = num_barcodes, 
                                                                    barcode_border = barcode_border, 
                                                                    final_width = final_size,
                                                                    final_height = final_size,
                                                                    word_count = 20)
            cv2.imwrite("{}/roi_{}_im/roi{}.png".format(output_dir, generated_type, generated+start_index), train)
            del train
            cv2.imwrite("{}/roi_{}_masks/roi_mask{}.png".format(output_dir, generated_type, generated+start_index), train_mask)
            del train_mask
            for i, im in enumerate(segmented_single_masks):
                cv2.imwrite("{}/roi_{}_masks/roi_mask{}_{}.jpg".format(output_dir, generated_type, generated+start_index, i), im)
            del segmented_single_masks
            bbxes['img_{:d}'.format(generated+start_index)] = {}
            bbxes['img_{:d}'.format(generated+start_index)]['boxes'] = all_bbx
        except AttributeError:
            # generated an image > IM_SIZE
            continue
        generated += 1
        pbar.update(1)
        # print("process", ind, "running")

    with open('temp{}.json'.format(int(ind)), 'w') as f:
        json.dump(bbxes, f)
    pbar.close()


if __name__ == '__main__':
    step = args.train_size // args.num_processes
    processes = []
    for i in range(args.num_processes):
        processes.append(mp.Process(target=generate, args=(i*step, (i+1)*step, args.start_index)))
        # processes.append(threading.Thread(target=generate, args=(i*step, (i+1)*step, args.start_index)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    all_bboxes = {}
    for i in range(args.num_processes):
        with open('temp{}.json'.format(i), 'r') as f:
            t = json.load(f)
            all_bboxes.update(t)
    with open('{}/roi_{}_masks/all_bboxes.json'.format(output_dir, generated_type), 'w') as f:
        json.dump(all_bboxes, f)

    print("Data generation finished.")