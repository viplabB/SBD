import multiprocessing as mp
import numpy as np
import argparse
from tqdm import tqdm
from generate_one_image import generate_one_training_image, DownsamplingImage
from PIL import Image

parser = argparse.ArgumentParser(description='Generate RPN proposals.')
parser.add_argument('--num_processes', '-n', type=int, default=6,
                    help='Number of processes used for image generation')
parser.add_argument('--train_size', '-t', type=int, default=10000,
                    help='Number of training images to be generated')
args = parser.parse_args()

num_barcodes_per_image = 10
barcode_border = 10
IM_SIZE = 2000

def generate(generated, train_size):
    pbar = tqdm(total=train_size-generated)
    while generated < train_size:
        try:
            train, train_mask = generate_one_training_image(num_barcodes = num_barcodes_per_image, 
                                                            barcode_border = barcode_border, 
                                                            final_size = IM_SIZE)
            im = Image.fromarray(train.astype(np.uint8))
            im.save("data_gen/X/train_{}.png".format(generated))
            im = Image.fromarray(DownsamplingImage(train_mask, down_sampling_factor = 4).astype(np.uint8))
            im.save("data_gen/Y/train_mask_{}.png".format(generated))
        except AttributeError:
            # generated an image > IM_SIZE
            continue
        generated += 1
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    step = args.train_size // args.num_processes
    processes = []

    for i in range(args.num_processes):
        processes.append(mp.Process(target=generate, args=(i*step, (i+1)*step,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Data generation finished.")