# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:55:43 2018

@author: Borja042
"""

# -*- coding: utf-8 -*-

#import libraries
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# Define function to calculate the delta
def uniScalarQuanti(sMax, sMin, L):
    return float((sMax-sMin)/L)

# Define function to quantize a signal value
def q(s, sMin, delta):
    return np.floor(((s-sMin)/delta)+0.5)*delta+sMin

# Read image with openCV
imgOriginal = cv2.imread('lena512.bmp',0)
imgTransformed = cv2.imread('lena512.bmp',0)

# Show the original image
print ("Original Image:")
imgplot = plt.imshow(imgTransformed,cmap='gray')
plt.show()

print("Size of the image:", imgOriginal[1].size, "x", imgOriginal[1].size)

#Obtain sMax from the image (maximum pixel value)
sMax=float(np.amax(imgOriginal))
print ("Maximum value of a pixel in the Original image=", sMax)

#Obtain sMin from the image (minimum pixel value)
sMin=float(np.amin(imgOriginal))
print ("Minimum value of a pixel in the Original image=", sMin)

print("_________________________________________________")


# Based on the quantization function defined above we will quantize and encode the original image
# Then we will compare the differences with the original image and measure the error for each possible R
# Also given L=2^R number of gray levels...we can estimate the probability of each level by making use of the histogram 
# And based on that we can compute the entropy for each R
R = 0
D_list = []
E_list = []
PSNR_list = []
for R in range(1,9):
    L = 2**R
    print ("R value ===>", R) 
    print("L value =", L)
    
    # Compute the delta
    delta=uniScalarQuanti(sMax, sMin, L)
    print ("Associated Delta =", delta)
    
    # Get all available gray levels for this R and L
    possible_levels=[(sMin+l*delta) for l in range(0, L+1)]

    # Plot the input/output characteristic function
    res_levels=[] 
    for i in range(2**8):
        res_levels=res_levels+[q(i, sMin, delta)]
    
    plt.step(range(2**8), res_levels)
    plt.title("Input/output function for R: "+str(R))
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()

    # Quantize the image and compute the histogram
    appearances = [0 for pl in possible_levels]
    for i in range(imgOriginal.shape[0]):
        for j in range(imgOriginal.shape[1]):
            pixel = imgOriginal.item(i, j)
            tpixel = q(pixel, sMin, delta)
            # Transform the image
            imgTransformed[i,j] = tpixel
            # Count for the histogram
            idx = possible_levels.index(tpixel)
            appearances[idx] = appearances[idx]+1
    
            
    # Plot the transformed image
    print ("Transformed Image:")
    imgplot = plt.imshow(imgTransformed,cmap='gray')
    plt.show()
    
    # Measure the distortion D, as the MSE of all signal values levels L
    df = pd.DataFrame({'original':np.array(imgOriginal.flatten(), dtype = 'f'), 'transformed':np.array(imgTransformed.flatten(), dtype = 'f')})

    #Apply Distorsion formula
    D = sum((df['original'] - df['transformed'])**2) / len(df)
    print ("Distortion with the original image =", D)
    
    #Save distorsion value for each Level L
    D_list += [D]
    
    # Store the PSNR index
    PSNR = 10 * math.log((255**2 / D), 10)
    print ("PSNR index with the original image =", PSNR)
    PSNR_list = PSNR_list + [PSNR]
    
    # Compute the entropy for L=2^R and save it
    E = 0
    for apps in appearances:
        prob_i = float(apps)/sum(appearances)
        if prob_i != 0:
            E = E - (prob_i * math.log(prob_i, 2))
    E_list = E_list + [E]
    print("Entropy associated to R and L =", E)
    
    print("_________________________________________________")

# Plot D as dependent of R
plt.plot(range(1,9), D_list)
plt.scatter(range(1,9), D_list,color='red')
plt.title("Distortion (D) for each possible value of R")
plt.xlabel("R value")
plt.ylabel("D(R value)")
plt.show()

print(D_list)

print("We can observe in the graph that the distortion decreases exponentially as we increase R. In particular there is a huge difference \
for R between 1 and 3, whereas for R bigger than 4 the distortion remains basically the same. This coincides with our observation of \
the transformed images, as they are losing less and less information while we increase the size of R.") 
print("The reason for that is that we are asking the encoder to use more bits to approximate the values in the image. By increasing R \
we are also increasing the number of possible values that the pixels can take, so the encoding becomes better. Indeed, with R=8 \
we should be able to get zero distortion, as we have space to represent all the original values as such, but due to the specificity \
of the Q(s) formula the encoded values do not always correspond with the original ones")

print("_________________________________________________")

# Entropy
# Plot Entropy as dependent of R
print(E_list)
plt.plot(range(1,9), E_list)
plt.scatter(range(1,9), E_list,color='red')
plt.title("Entropy (E) for each possible value of R")
plt.xlabel("R value")
plt.ylabel("Entropy(R value)")
plt.show()

print("We can observe how the quantity of information available, in terms of the Shanon entropy in this case, \
increases as we increase the value of R. This is clear if we consider that the higher the number of bits, \
the higher the value of L and the amount of information that we can represent.")

print("_________________________________________________")

# Peak Signal-to-Noise Ratio
# PSNR as dependent of R, based on the mse
# Plot Entropy as dependent of R
plt.plot(range(1,9), PSNR_list)
plt.scatter(range(1,9), PSNR_list,color='red')
plt.title("PSNR for each possible value of R")
plt.xlabel("R value")
plt.ylabel("PSNR(R value)")
plt.show()

print("In this occasion we are measuring the quality of a signal with regards to the noise introduced on it \
considering the conditions of the signal such as the maximum number of bits. \
Clearly the PSNR increases with the number of bits, as the noise introduced becomes smaller. \
In fact, PSNR is often used to measure the visualization quality of the images. \
Its interesting though how the index seems very similar for 7 and 8 bits, where the images are \
almost indistinguishable for the human eye.")
