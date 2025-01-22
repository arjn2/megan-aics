# megan-aics

Enhanced Style-GAN based training model which extends dataset by generating new variants without compromising quality of generation.

Combined encoders with styleGAN for quality generation for new images.

# Dataset used

Malevis balanced dataset from kaggle is modified with new class of additionally added benign pe images.

This custom benign class will reduce false positives.

dataset/pe2img.py is the python script used to convert pe files to gray images.


# Optimizations

Well optimized for both gpu and cpu ,can run easily on google colab and local system also.

Strict paramters for memory efficiency, Avoid overfitting using early stopping patience, batch file optimizations ,etc.

improved training speed on GPU.

run well with 7-8 gb vram lock.

# Improvement using CONFUSION MATRIX

The below show comparison of normal real dataset vs mixed dataset (real+generated) both are trained compared using cnn.

Mixed (Real+generated data) MODEL 
![GAN-model](https://github.com/arjn2/megan-aics/blob/main/conf-matrix/gan-model.png?raw=true)

MODEL trained only in Real dataset 
![NON-GAN](https://github.com/arjn2/megan-aics/blob/main/conf-matrix/non-gan.png?raw=true)

# References
```
https://github.com/arjn2/MalDICT-reference
Dikedataset benign github
```
