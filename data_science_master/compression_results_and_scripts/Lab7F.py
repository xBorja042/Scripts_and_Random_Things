#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:01:19 2018

@author: fran
"""

import datetime
import imutils
import cv2
import numpy as np
import pandas as pd
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import imageio
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *

import time
from os import listdir
from os.path import isfile, join
import math
import matplotlib.pyplot as plt
import huffman


def mse_frames(frame1, frame2):
    return ((frame1 - frame2) ** 2).mean(axis=None)


def mae_frames(frame1, frame2):
    df_error = pd.DataFrame(
        {'original': np.array(frame1.flatten(), dtype='f'), 'transformed': np.array(frame2.flatten(), dtype='f')})
    df_error['error'] = (df_error['original'] - df_error['transformed']).abs()
    return sum(df_error['error']) / len(df_error)


# Define function to calculate the delta
def uniScalarQuanti(sMax, sMin, L):
    return float((sMax - sMin) / L)


# Define function to quantize a signal value
def q(s, sMin, delta):
    if delta != 0:
        return np.floor(((s - sMin) / delta) + 0.5) * delta + sMin
    else:  # If all values of image were the same
        return np.floor(s)


def quantization(img):
    R = 0
    delta_list = []
    distortion_list = []
    shannon_entropy_list = []
    huffman_entropy_list = []
    PSNR_list = []
    img_list = []

    imgtransf = np.zeros(shape=(len(img), len(img[0])))
    sMax = float(np.amax(img))
    sMin = float(np.amin(img))

    # if sMax == sMin:
    #    print("Image has only one value, no point on quantize it!")
    #    return delta_list, distortion_list, PSNR_list, shannon_entropy_list, huffman_entropy_list, img_list

    for R in range(1, 9):
        L = 2 ** R
        # print ("R value ===>", R)
        # print("L value =", L)

        # Compute the delta
        delta = uniScalarQuanti(sMax, sMin, L)
        delta_list = delta_list + [delta]
        # print ("Associated Delta =", delta)

        # Get all available gray levels for this R and L
        possible_levels = [(sMin + l * delta) for l in range(0, L + 1)]

        # Quantize the image and compute the histogram
        appearances = [0 for pl in possible_levels]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img.item(i, j)
                tpixel = q(pixel, sMin, delta)
                # Transform the image
                imgtransf[i, j] = tpixel
                # Count for the histogram
                idx = possible_levels.index(tpixel)
                appearances[idx] = appearances[idx] + 1
                # Plot the transformed image
                #        print ("Transformed Image:")
                #        imgplot = plt.imshow(imgtransf,cmap='gray')
                #        plt.show()
        img_list = img_list + [imgtransf.copy()]

        # Measure the distortion D, as the MSE of all signal values
        df_error = pd.DataFrame(
            {'original': np.array(img.flatten(), dtype='f'), 'transformed': np.array(imgtransf.flatten(), dtype='f')})
        df_error['error'] = (df_error['original'] - df_error['transformed']) ** 2

        distortion = sum(df_error['error']) / len(df_error)
        # print ("Distortion with the original image =", distortion)
        distortion_list += [distortion]

        # Store the PSNR index
        if distortion != 0:
            PSNR = 10 * math.log((255 ** 2 / distortion), 10)
        else:
            PSNR = float("inf")
        # print ("PSNR index with the original image =", PSNR)
        PSNR_list = PSNR_list + [PSNR]

        # Compute the entropy for L=2^R and save it
        entropy = 0
        for apps in appearances:
            prob_i = apps / sum(appearances)
            if prob_i != 0:
                entropy = entropy - (prob_i * math.log(prob_i, 2))
        shannon_entropy_list = shannon_entropy_list + [entropy]
        # print("Shannon entropy associated to R and L =", entropy)

        unique, counts = np.unique(imgtransf, return_counts=True)
        ocurrences = [[unique[i], counts[i]] for i in range(len(unique))]
        codes = huffman.codebook(ocurrences)
        avg_length = 0
        for i in range(len(unique)):
            # print(len(codes[unique[i]]))
            avg_length += counts[i] / imgtransf.size * len(codes[unique[i]])
        huffman_entropy_list = huffman_entropy_list + [avg_length]
        # print("Huffman entropy associated to R and L =", avg_length)

        # print("_________________________________________________")

    return delta_list, distortion_list, PSNR_list, shannon_entropy_list, huffman_entropy_list, img_list


def decomposition(img, max_levels=2):
    # Decomposition
    next_level = np.array(img)
    fig = plt.figure()
    for k in range(1, max_levels + 1):

        # Operamos por filas
        decomp_matrix = []
        for line in next_level:
            means = []
            diffs = []
            for i in range(0, line.size, 2):
                mean = float(np.average(line[i:i + 2]))
                means = means + [mean]
                diffs = diffs + [line[i] - mean]
            decomp_matrix = decomp_matrix + [means + diffs]

        # Operamos por columnas (para ello hacemos la transp, operamos por filas y hacemos transp del resultado)
        transp = np.array(decomp_matrix).transpose()
        decomp_matrix2 = []
        for line in transp:
            means = []
            diffs = []
            for i in range(0, line.size, 2):
                mean = float(np.average(line[i:i + 2]))
                means = means + [mean]
                diffs = diffs + [line[i] - mean]
            decomp_matrix2 = decomp_matrix2 + [means + diffs]
        decomp_matrix2 = np.array(decomp_matrix2).transpose()

        h = len(decomp_matrix2)
        w = len(decomp_matrix2[0])
        next_level = np.array(decomp_matrix2[:int(h / 2), :int(w / 2)])
        if k != 1:
            last_matrix[:h, :w] = np.array(decomp_matrix2)
        else:
            last_matrix = np.array(decomp_matrix2)
        del (decomp_matrix, decomp_matrix2)
    return last_matrix


def recomposition(img, max_levels=2):
    decomposed = np.array(img)
    next_level = np.array(
        decomposed[:int(len(decomposed) / (2 ** (max_levels - 1))), :int(len(decomposed[0]) / (2 ** (max_levels - 1)))])

    for k in range(max_levels, 0, -1):

        # Operamos por columnas (para ello hacemos la transp, operamos por filas y hacemos transp del resultado)
        transp = np.array(next_level).transpose()
        recomp_matrix = []
        for line in transp:
            orig = []
            for i in range(0, int(line.size / 2)):
                mean = line[i]
                dif = line[i + int(line.size / 2)]
                v1 = mean + dif
                v2 = mean - dif
                orig = orig + [v1, v2]
            recomp_matrix = recomp_matrix + [orig]
        recomp_matrix = np.array(recomp_matrix).transpose()

        # Operamos por filas
        recomp_matrix2 = []
        for line in recomp_matrix:
            orig = []
            for i in range(0, int(line.size / 2)):
                mean = line[i]
                dif = line[i + int(line.size / 2)]
                v1 = mean + dif
                v2 = mean - dif
                orig = orig + [v1, v2]
            recomp_matrix2 = recomp_matrix2 + [orig]

        h = len(recomp_matrix2)
        w = len(recomp_matrix2[0])
        decomposed[:h, :w] = recomp_matrix2
        next_level = np.array(decomposed[:2 * h, :2 * w])
        next_level[:h, :w] = recomp_matrix2

    reconstructed = np.array(decomposed)
    return reconstructed


# Read the video
#path = "./"
vname="mono.mp4"
vpath="C:\\Users\\Borja042\\Desktop\\MASTER2\\COMPRESION\\LABS567\\OUT6"
video = vname

print(video)
camera = cv2.VideoCapture(video)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frames_width = int(camera.get(3))
frames_height = int(camera.get(4))

print("Video width in px:", frames_width)
print("Video height in px:", frames_height)

m = 16
p = 1

nblocks_width = int(frames_width / m)
nblocks_height = int(frames_height / m)

print("Block size:", m)
print("Pixels for search around:", p)

print("Number of horizontal blocks:", nblocks_width)
print("Number of vertical blocks:", nblocks_height)

# Choose quantization level
chosen_R = 5

# Grab first frame
(grabbed, frame_curr) = camera.read()
frame_ref_gray = cv2.cvtColor(frame_curr.copy(), cv2.COLOR_BGR2GRAY)

# Loop over the frames of the video
i = 1
n = 5
orig_frames_dif_list = []
avg_motion_compensated_error_list = []
avg_distortion_list = []
huffman_frames_list = []
while i < n:

    # Grab another frame
    (grabbed, frame_curr) = camera.read()
    i = i + 1

    print("Frame: " + str(i))

    # If the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break

    # Convert to gray and get a copy for printing output
    frame_curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    copy = cv2.cvtColor(frame_curr_gray.copy(), cv2.COLOR_GRAY2RGB)

    # For all the blocks in the grid look for its best match in reference
    motion_vectors = []
    for h in range(nblocks_height):
        motion_vectors_h = []
        for w in range(nblocks_width):
            # Store block position
            y1 = h * m  # h/w refers to the block number
            y2 = y1 + m  # y/x refers to the pixel in image (cartesian axes style)
            x1 = w * m
            x2 = x1 + m
            block = frame_curr_gray[y1:y2, x1:x2]
            # Compare with surroundings (start with min mse as the one with same position)
            min_mse = mse_frames(block, frame_ref_gray[y1:y2, x1:x2])
            best_y_mov = 0
            best_x_mov = 0
            # Check all surroundings areas in reference to get the one with smallest MSE, full search method
            for ref_y1 in range(max(y1 - p, 0), min(y1 + p, frames_height - m) + 1):
                for ref_x1 in range(max(x1 - p, 0), min(x1 + p, frames_width - m) + 1):
                    mse = mse_frames(block, frame_ref_gray[ref_y1:ref_y1 + m, ref_x1:ref_x1 + m])
                    if mse < min_mse:
                        min_mse = mse
                        best_y_mov = y1 - ref_y1
                        best_x_mov = x1 - ref_x1
            # Save motion vector
            motion_vectors_h += [[best_x_mov, best_y_mov]]
        motion_vectors += [motion_vectors_h]
    print("Block matching finished!")

    # Compute the motion compensated frame
    frame_motion_comp = frame_ref_gray.copy()
    # For all the blocks in the grid reposition them based on the motion vectors
    for h in range(nblocks_height):
        for w in range(nblocks_width):
            # Store block position
            y1 = h * m  # h/w refers to the block number
            y2 = y1 + m  # y/x refers to the pixel in image (cartesian axes style)
            x1 = w * m
            x2 = x1 + m
            mv = motion_vectors[h][w]
            # print(h, w, mv, x1, x2, y1, )
            y_mov = mv[1]
            x_mov = mv[0]
            frame_motion_comp[y1:y2, x1:x2] = frame_ref_gray[y1 - y_mov:y2 - y_mov, x1 - x_mov:x2 - x_mov]

    # Compute the movement measure and the average motion compensated error
    # orig_frames_dif = mae_frames(frame_curr_gray, frame_ref_gray)
    # orig_frames_dif_list += [orig_frames_dif]
    # print("Difference original:", orig_frames_dif)
    # avg_motion_compensated_error = mae_frames(frame_curr_gray, frame_motion_comp)
    # avg_motion_compensated_error_list += [avg_motion_compensated_error]
    # print("Avg motion compensated error:", avg_motion_compensated_error)

    # Compute the difference btw current and motion compensated and encode it
    Eres = frame_curr_gray - frame_motion_comp
    decomp_eres = decomposition(Eres, max_levels=2)
    delta_list, distortion_list, PSNR_list, shannon_entropy_list, huffman_entropy_list, img_list = quantization(
        decomp_eres)
    quant = img_list[chosen_R - 1]
    # Next step would be Huffman coding but we will skip it (unless we need to actually store the encoded version)
    # cv2.imwrite(vpath+"Encoded outputs\\"+str(i)+"_encoded_eres.png", quant)
    # Reconstruct Eres from encoded version and predict the current frame unsing the previous one, the motion vectors and Eres
    recomp_eres = recomposition(quant, max_levels=2)
    frame_curr_predicted = frame_motion_comp + recomp_eres

    plt.imshow(frame_curr_gray, cmap='gray')

    plt.title("Original")
    plt.show()

    plt.imshow(frame_curr_predicted, cmap='gray')
    plt.title("Predicted")
    plt.show()

    # Some information
    huffman_frame = huffman_entropy_list[chosen_R - 1]
    huffman_frames_list += [huffman_frame]
    print("Huffman entropy for the Eres:", huffman_frame)
    # Compute the distortion between the original frame and the predicted one
    avg_distortion = mse_frames(frame_curr_gray, frame_curr_predicted)
    avg_distortion_list += [avg_distortion]
    print("Avg distortion (diff between original and predicted):", avg_distortion)

    # Keep prediction for next
    frame_ref_gray = frame_curr_predicted.copy()

    print("__________________________")

# Compute the distortion between the reconstructed frames and original ones.
# Subsequently compute the total average distortion (related tothe total number of frames) of the video.
print("Total Huffman entropy:", np.mean(huffman_frames_list))
print("Total average distortion:", np.mean(avg_distortion_list))

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

# Read the video
path = "./"
vpath = "./"
vname = "video.mp4"
video = vpath + vname

start_time = 0
end_time = 3
ffmpeg_extract_subclip("mono.mp4", start_time, end_time, targetname="trim_video.mp4")

trime_video = "trim_video.mp4"

"""• Plot a graph of the total average distortion vs. video bitrate (nr. of
bits in unit of time, e.g Kbps or Mbps), for different quantization steps
of the wavelet subbands. Comment your results.
• Plot a graph of the PSNR (considering the total average distortion) vs.
video bitrate, for different quantization steps of the wavelet subbands.
Comment your results.
"""
### SACAR EL VIDEO BIT RATE POR ALGUN LADO
### REPETIR TOOOODO PARA DIFFERENT QUANTIZATION STEPS (chosen_R)



# Choose quantization level
chosen_R = 0

for z in range(1, 9):

    print("_____________________________")

    print(trime_video)
    camera = cv2.VideoCapture(trime_video)

    chosen_R = chosen_R + 1
    print("Chosen R: " + str(chosen_R))

    fps = camera.get(cv2.CAP_PROP_FPS)
    print("FPS: ", fps)

    bitrate = frames_width * frames_height * chosen_R * fps

    print("bitrate for R ", chosen_R, "= ", bitrate)
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frames_width = int(camera.get(3))
    frames_height = int(camera.get(4))

    print("Video width in px:", frames_width)
    print("Video height in px:", frames_height)

    m = 16
    p = 4

    nblocks_width = int(frames_width / m)
    nblocks_height = int(frames_height / m)

    print("Block size:", m)
    print("Pixels for search around:", p)

    print("Number of horizontal blocks:", nblocks_width)
    print("Number of vertical blocks:", nblocks_height)

    # Grab first frame
    (grabbed, frame_curr) = camera.read()
    frame_ref_gray = cv2.cvtColor(frame_curr.copy(), cv2.COLOR_BGR2GRAY)

    # Loop over the frames of the video
    i = 1
    n = 5
    orig_frames_dif_list = []
    avg_motion_compensated_error_list = []
    avg_distortion_list = []
    huffman_frames_list = []
    while i < n:

        # Grab another frame
        (grabbed, frame_curr) = camera.read()
        i = i + 1

        print("Frame: " + str(i))

        # If the frame could not be grabbed, then we have reached the end of the video
        if not grabbed:
            break

        # Convert to gray and get a copy for printing output
        frame_curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        copy = cv2.cvtColor(frame_curr_gray.copy(), cv2.COLOR_GRAY2RGB)

        # For all the blocks in the grid look for its best match in reference
        motion_vectors = []
        for h in range(nblocks_height):
            motion_vectors_h = []
            for w in range(nblocks_width):
                # Store block position
                y1 = h * m  # h/w refers to the block number
                y2 = y1 + m  # y/x refers to the pixel in image (cartesian axes style)
                x1 = w * m
                x2 = x1 + m
                block = frame_curr_gray[y1:y2, x1:x2]
                # Compare with surroundings (start with min mse as the one with same position)
                min_mse = mse_frames(block, frame_ref_gray[y1:y2, x1:x2])
                best_y_mov = 0
                best_x_mov = 0
                # Check all surroundings areas in reference to get the one with smallest MSE, full search method
                for ref_y1 in range(max(y1 - p, 0), min(y1 + p, frames_height - m) + 1):
                    for ref_x1 in range(max(x1 - p, 0), min(x1 + p, frames_width - m) + 1):
                        mse = mse_frames(block, frame_ref_gray[ref_y1:ref_y1 + m, ref_x1:ref_x1 + m])
                        if mse < min_mse:
                            min_mse = mse
                            best_y_mov = y1 - ref_y1
                            best_x_mov = x1 - ref_x1
                # Save motion vector
                motion_vectors_h += [[best_x_mov, best_y_mov]]
            motion_vectors += [motion_vectors_h]
        print("Block matching finished!")

        # Compute the motion compensated frame
        frame_motion_comp = frame_ref_gray.copy()
        # For all the blocks in the grid reposition them based on the motion vectors
        for h in range(nblocks_height):
            for w in range(nblocks_width):
                # Store block position
                y1 = h * m  # h/w refers to the block number
                y2 = y1 + m  # y/x refers to the pixel in image (cartesian axes style)
                x1 = w * m
                x2 = x1 + m
                mv = motion_vectors[h][w]
                # print(h, w, mv, x1, x2, y1, )
                y_mov = mv[1]
                x_mov = mv[0]
                frame_motion_comp[y1:y2, x1:x2] = frame_ref_gray[y1 - y_mov:y2 - y_mov, x1 - x_mov:x2 - x_mov]

        # Compute the movement measure and the average motion compensated error
        # orig_frames_dif = mae_frames(frame_curr_gray, frame_ref_gray)
        # orig_frames_dif_list += [orig_frames_dif]
        # print("Difference original:", orig_frames_dif)
        # avg_motion_compensated_error = mae_frames(frame_curr_gray, frame_motion_comp)
        # avg_motion_compensated_error_list += [avg_motion_compensated_error]
        # print("Avg motion compensated error:", avg_motion_compensated_error)

        # Compute the difference btw current and motion compensated and encode it
        Eres = frame_curr_gray - frame_motion_comp
        decomp_eres = decomposition(Eres, max_levels=2)
        delta_list, distortion_list, PSNR_list, shannon_entropy_list, huffman_entropy_list, img_list = quantization(
            decomp_eres)
        quant = img_list[chosen_R - 1]
        # Next step would be Huffman coding but we will skip it (unless we need to actually store the encoded version)
        # cv2.imwrite(vpath+"Encoded outputs\\"+str(i)+"_encoded_eres.png", quant)
        # Reconstruct Eres from encoded version and predict the current frame unsing the previous one, the motion vectors and Eres
        recomp_eres = recomposition(quant, max_levels=2)
        frame_curr_predicted = frame_motion_comp + recomp_eres

        plt.imshow(frame_curr_gray, cmap='gray')

        plt.title("Original")
        plt.show()

        plt.imshow(frame_curr_predicted, cmap='gray')
        plt.title("Predicted")
        plt.show()

        # Some information
        huffman_frame = huffman_entropy_list[chosen_R - 1]
        huffman_frames_list += [huffman_frame]
        print("Huffman entropy for the Eres:", huffman_frame)
        # Compute the distortion between the original frame and the predicted one
        avg_distortion = mse_frames(frame_curr_gray, frame_curr_predicted)
        avg_distortion_list += [avg_distortion]
        print("Avg distortion (diff between original and predicted):", avg_distortion)

        # Keep prediction for next
        frame_ref_gray = frame_curr_predicted.copy()

        print("__________________________")

    # Compute the distortion between the reconstructed frames and original ones.
    # Subsequently compute the total average distortion (related tothe total number of frames) of the video.
    print("Total Huffman entropy:", np.mean(huffman_frames_list))
    print("Total average distortion:", np.mean(avg_distortion_list))

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()