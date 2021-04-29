import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud
from random_words import RandomWords
from PIL import Image
import glob
# np.random.seed(42)

import pickle
clean_barcodes = []
for i in tqdm(range(36)):
    with open("clean_barcodes_{}.pickle".format(i), 'rb') as f:
        clean_barcodes += pickle.load(f)

def UpsamplingImage(img, up_sampling_factor = 2):
    w = int(img.shape[1]*up_sampling_factor)
    h = int(img.shape[0]*up_sampling_factor)
    img_up = np.array(Image.fromarray(img).resize((w, h), resample=Image.BILINEAR))
    return img_up

def show_images_overlayed_single (Im1, Im1_mask, title1=""):
    a = 0.5 #transparency parameter
    
    plt.figure(figsize=[10, 20])
    plt.subplot(1, 2, 1)
    plt.imshow(Im1, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(Im1_mask, cmap="gray")
    
    plt.figure(figsize=[10, 10])
    plt.imshow(Im1, cmap="gray")
    plt.imshow(Im1_mask, alpha=a)

def overlay(src, dest, upper_left):
    x_start = upper_left[0]
    x_end = upper_left[0] + src.shape[0]
    y_start = upper_left[1]
    y_end = upper_left[1] + src.shape[1]
    dest[x_start:x_end, y_start:y_end] = src
    return dest

def rotateImage(img, angle, borderValue=255):
    (h, w) = img.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT,flags = cv2.INTER_AREA, borderValue=borderValue)

# warp img2 to img1 with homograph H
def warpTwoImages(img1, img2, H, borderValue=255):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin), borderMode=cv2.BORDER_CONSTANT, borderValue = borderValue)
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def HomographyImage(img, x_bl, x_br, borderValue=255):
    h, w = img.shape
    pts_src = np.array([[0, 0],     #top left
                        [0, w],     #top right
                        [h, 0],     #bottom left
                        [h, w]])    #bottom right

    pts_dst = np.array([[0, 0],     #top left
                        [0, w],     #top right
                        [x_bl, 0],  #bottom left
                        [x_br, w]]) #bottom right
    
    H_mat, status = cv2.findHomography(pts_src, pts_dst)
    out_hom_temp = cv2.warpPerspective(img, H_mat, (h,w), borderMode=cv2.BORDER_CONSTANT, borderValue = borderValue) 
    return warpTwoImages(out_hom_temp, img, H_mat, borderValue)

def ImEnlarger(im, Max_height, Max_width, borderValue=255):
    
    #print(im.shape[0], im.shape[1])
    row, col = im.shape
    if row > Max_height or col > Max_width:
        raise AttributeError("image is already larger than Max_size")
    
    tpad_ = int(Max_height - row) // 2
    bpad_ = int(Max_height - row) - tpad_
    lpad_ = int(Max_width - col) // 2
    rpad_ = int(Max_width - col) - lpad_
        
    padded = cv2.copyMakeBorder(im, tpad_, bpad_, lpad_, rpad_, \
                                     cv2.BORDER_CONSTANT, value=borderValue)   
 
    return padded

def DownsamplingImage(img, down_sampling_factor = 2):
    w = int(img.shape[1]//down_sampling_factor)
    h = int(img.shape[0]//down_sampling_factor)
    img_down = np.array(Image.fromarray((img).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR))
    return img_down

def generateRandomWords(word_count = 120):
    random_words = RandomWords().random_words(count=word_count)
    for word in random_words[:word_count//3]:
        word.capitalize()
    for word in random_words[word_count//3:word_count//3*2]:
        word.upper()
    random_numbers = [str(num) for num in np.random.randint(1000, 10000000, size=word_count//3)]
    words = ' '.join(random_words + random_numbers)
    return words
"""
all_empty_images = []
print("loading empty images")
for im_name in tqdm(glob.glob("./empty_images/*.jpg")):
    all_empty_images.append(cv2.imread(im_name))

def generate_random_background(background_shape):
    rand_im = np.random.choice(all_empty_images)
    h, w, c = rand_im.shape
    max_h, max_w = h - background_shape[0], w - background_shape[1]
    rand_h, rand_w = np.random.randint(0, max_h), np.random.randint(0, max_w)

    return rand_im[rand_h:rand_h+background_shape[0], rand_w:rand_w+background_shape[1]]
"""

barcode_index = 0
def generate_one_training_image(num_barcodes = 5, barcode_border = 10, final_width = 2000, final_height = 2000, word_count = 120):
    words = generateRandomWords(word_count=word_count)
    image_width = final_width * 17 // 30
    image_height = final_height * 17 // 30
    image_size = min(image_width, image_height)
    background = WordCloud(width=image_width, height=image_height, min_font_size=1, max_font_size=image_size*4//100,\
                      max_words=image_size//10, background_color="white", \
                      color_func=lambda *args, **kwargs: "black").generate(words).to_array()
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    #background = WordCloud(width=image_width, height=image_height, min_font_size=1, max_font_size=image_size*4//100,\
    #                  max_words=image_size//10, background_color=(1, 1, 1), \
    #                  color_func=lambda *args, **kwargs: (0, 0, 0)).generate(words).to_array()
    #im = generate_random_background((image_height, image_width))
    #background = np.multiply(im, background)
    mask_background_no_margin = np.zeros(background.shape)
    all_single_barcode_masks = []
    
    # paste barcodes on background
    for i in range(num_barcodes):
        global barcode_index
        barcode = clean_barcodes[barcode_index % len(clean_barcodes)]
        barcode_index += 1
        barcode = barcode.astype(np.uint8)
        barcode = UpsamplingImage(barcode, (final_width / 2000))
        # barcode = cv2.cvtColor(barcode, cv2.COLOR_GRAY2RGB)
        margined_barcode = cv2.copyMakeBorder(barcode, \
                                              barcode_border, barcode_border, barcode_border, barcode_border,\
                                              cv2.BORDER_CONSTANT, value=255)
        # paste one barcode
        paste_barcode_try = 0 # in case there is no possible no-overlap place
        while (paste_barcode_try < 100):
            x, y = np.random.randint(barcode_border, background.shape[1] - margined_barcode.shape[1]), np.random.randint(barcode_border, background.shape[0] - margined_barcode.shape[0])
            upper_left = (y, x)
            # check if target area contains any 255
            if mask_background_no_margin[y-barcode_border:y+margined_barcode.shape[0],\
                                         x-barcode_border:x+margined_barcode.shape[1]].max() > 0:
                paste_barcode_try += 1
                continue
            background = overlay(margined_barcode, background, upper_left)
            upper_left_no_margin = (upper_left[0] + barcode_border, upper_left[1] + barcode_border)
            mask_background_no_margin = overlay(np.ones(barcode.shape) * 255, mask_background_no_margin, upper_left_no_margin)
            single_barcode_mask = overlay(np.ones(barcode.shape) * 255, np.zeros(background.shape), upper_left_no_margin)
            all_single_barcode_masks.append(single_barcode_mask)
            break
    # rotate and homography
    rotation_angle = np.random.randint(0, 360)
    homography_dst = (np.random.randint(background.shape[0], background.shape[0]*3//2), \
                      np.random.randint(background.shape[0], background.shape[0]*3//2))
    background = rotateImage(HomographyImage(background, homography_dst[0], homography_dst[1]), rotation_angle)
    mask_background_no_margin = rotateImage(HomographyImage(mask_background_no_margin, homography_dst[0], 
                                                            homography_dst[1], borderValue=0), 
                                            rotation_angle, borderValue=0)
    all_single_barcode_masks = [rotateImage(HomographyImage(single_barcode_mask, homography_dst[0], 
                                                            homography_dst[1], borderValue=0), 
                                            rotation_angle, borderValue=0) for single_barcode_mask in all_single_barcode_masks]
    # enlarge and generate dark border
    border_color = np.random.randint(0, 200)
    background = ImEnlarger(background, final_height, final_width, borderValue=border_color)
    mask_background_no_margin = ImEnlarger(mask_background_no_margin, final_height, final_width, borderValue=0)
    all_single_barcode_masks = [ImEnlarger(single_barcode_mask, final_height, final_width, borderValue=0) for single_barcode_mask in all_single_barcode_masks]

    # darken
    darkenFactor = np.clip(np.random.normal(0.9, 0.2), 0.5, 1.0)
    background = darkenFactor*background.astype(np.float)

    # find bounding boxes
    all_bbx = []
    for im in all_single_barcode_masks:
        contours, _ = cv2.findContours(im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0 
        x,y,w,h = cv2.boundingRect(contours[0])
        all_bbx.append([x,y,x+w,y+h])
    return background, mask_background_no_margin, all_single_barcode_masks, all_bbx
