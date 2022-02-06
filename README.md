# Synthetic Barcode Datasets (SBD)

## Overview
This barcode datasets contains:

- 100,000 LR synthetic barcode datasets along with their corresponding bounding boxes ground truth masks.
- 100,000 UHR synthetic barcode datasets along with along with their corresponding bounding boxes ground truth masks.

## Size
The LR datasets (~2.2G) could be downloaded here at [BarcodesLR](https://drive.google.com/file/d/1q9AU1y9qs2yaAPe1K2a8Th5AFKm03_uZ/view?usp=sharing). It constitutes images of resolution 400 x 400 px.

The UHR datasets (~150G) could be downloaded in 15G increments at [BarcodesUHR00](https://drive.google.com/file/d/1Mxb3z9VnE-SRTlq4nqgPzeyrdxXH4hlj/view?usp=sharing), [BarcodesUHR01](https://drive.google.com/file/d/18WQEHoiba7eHeoLtlkFvdIsurm3tTJNs/view?usp=sharing), [BarcodesUHR02](https://drive.google.com/file/d/1IrhRu_SKewKbvW6ozk4GqNzrzCX3SzkX/view?usp=sharing), [BarcodesUHR03](https://drive.google.com/file/d/1nPbxn18Ex0DIQDc7rxs6UrC4nJ_gLqln/view?usp=sharing), [BarcodesUHR04](https://drive.google.com/file/d/1nMKhPPv-Lyz0eAKgKCPNHUbsOMAyDPND/view?usp=sharing), [BarcodesUHR05](https://drive.google.com/file/d/106wr_Fmeayr0XQQJ54detgTPEe1T88E5/view?usp=sharing), [BarcodesUHR06](https://drive.google.com/file/d/1wxRrud2w8mI1wu5_dra6H5wXnBHlx-ya/view?usp=sharing), [BarcodesUHR07](https://drive.google.com/file/d/17S7SwIvub04My7OlXaYSY407ArN2dHIo/view?usp=sharing), [BarcodesUHR08](https://drive.google.com/file/d/1M5JYbC_EXOeNm1mS-gBUypjqNlMOgquh/view?usp=sharing), [BarcodesUHR09](https://drive.google.com/file/d/1KFmsr0P-YleN2q9YRmTGUZl7tJIpHUZ-/view?usp=sharing), and recombined. It constitutes images of resolution ~ >= 10k x 10k px.

## How to recombine UHR datasts images
1. Download all the individual UHR files into the same directory (ex. myDirectory/).
2. In terminal, navigate the directory location and run the Linux command: "cat UHR* > UHR.tar.gz".
3. Decompress the recombined file with the Linux command "tar -xvzf UHR.tar.gz -C myDatasetsDirectory".



## Types of barcodes
Code 39, Code 93, Code 128, UPC, EAN, PD417, ITF, Data Matrix, AZTEC, and QR among others.

## Reference
The paper titled "Fast, Accurate Barcode Detection in Ultra High-Resolution Images" by Quenum et al. introducing this dataset was accepted in IEEE International Conference on Image Processing, September 2021, USA and coud be found [here](https://ieeexplore.ieee.org/document/9506134).

## Sample synthesized UHR barcode image
![train_124](https://user-images.githubusercontent.com/82744965/115137026-8f987480-9ff1-11eb-8628-d47f54d622d2.png)

## Citation
J. Quenum, K. Wang and A. Zakhor, "Fast, Accurate Barcode Detection in Ultra High-Resolution Images," 2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 1019-1023, doi: 10.1109/ICIP42928.2021.9506134.


