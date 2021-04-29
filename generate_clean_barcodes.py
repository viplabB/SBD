from tqdm import tqdm
import pickle
import numpy as np
import treepoem
import string
import random
import argparse
import multiprocessing as mp

parser = argparse.ArgumentParser(description='Generate clean barcodes.')
parser.add_argument('--num_processes', '-n', type=int, default=6,
                    help='Number of processes used for image generation')
parser.add_argument('--size', '-s', type=int, default=30000,
                    help='Number of training images to be generated')
args = parser.parse_args()

def createCleanBarcode(codeType='random'):
    barcode_types = {
        'upca':{'size':11, 'H':146, 'W':190},
        'ean13':{'size':12, 'H':146, 'W':190},
        'qrcode':{'size':50, 'H':102, 'W':102},
        'pdf417':{'size':50, 'H':62, 'W':242}
    }
    if codeType == 'random':
        codeType = np.random.choice(list(barcode_types))
        
    letters = string.digits
    data = ''.join(random.choice(letters) for i in range(barcode_types[codeType]['size']))
    
    generated = treepoem.generate_barcode(barcode_type=codeType, data=data).convert('1')
    # convert PIL Image to array
    arr = np.array(generated).astype(np.int)
    arr[arr==1] = 255
    return arr

def generate(ind, length):
    clean_barcodes = []
    for i in tqdm(range(length)):
        clean_barcodes.append(createCleanBarcode())

    with open("clean_barcodes_{}.pickle".format(ind), 'wb') as f:
        pickle.dump(clean_barcodes, f)
    
if __name__ == '__main__':
    step = args.size // args.num_processes
    processes = []

    for i in range(args.num_processes):
        processes.append(mp.Process(target=generate, args=(i, step, )))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    all_barcodes = []
    for i in range(args.num_processes):
        with open("clean_barcodes_{}.pickle".format(i), 'rb') as f:
            all_barcodes += pickle.load(f)
            
    with open("clean_barcodes.pickle", 'wb') as f:
        pickle.dump(all_barcodes, f)
    

    print("Data generation finished.")