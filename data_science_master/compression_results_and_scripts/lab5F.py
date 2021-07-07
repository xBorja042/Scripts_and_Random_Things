# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:36:28 2018

@author: Borja042
"""

import datetime
#import imutils
import cv2
import numpy as np
import pandas as pd
#from imutils.object_detection import non_max_suppression
#from tqdm import tqdm
import time
from os import listdir
from os.path import isfile, join

def mse_frames(frame1, frame2):
    df_error = pd.DataFrame({'original':np.array(frame1.flatten(), dtype = 'f'), 'transformed':np.array(frame2.flatten(), dtype = 'f')})
    df_error['error'] = (df_error['original'] - df_error['transformed'])**2
    return sum(df_error['error']) / len(df_error)

#path="C:\\Users\\ricardo\\Desktop\\Niza\\PrimerSemestre\\Compression\\Labs\\"
#vpath="C:\\Users\\ricardo\\Desktop\\Niza\\PrimerSemestre\\Compression\\Labs\\"
vname="mono.mp4"
vpath="\\OUT5\\"
video=vname
print(video)

camera = cv2.VideoCapture(video)
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
# Loop over the frames of the video until we get the one we want
# Loop over the frames of the video until we get the one we want
i = 1
n = 10
while i<=n:
    
    # Grab another frame
    (grabbed, frame_curr) = camera.read()
 
    print("Frame: "+str(i))   

    # If the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break
    # Keep frame number n-1 for reference and convert to gray
    if i == n-1:
        frame_ref_gray = cv2.cvtColor(frame_curr.copy(), cv2.COLOR_BGR2GRAY)
    
    # Ignore following until frame number becomes n
    if  i != n:
        i=i+1
        continue
    i=i+1
        # Convert to gray and get a copy for printing output    
    frame_curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    copy = cv2.cvtColor(frame_curr_gray.copy(), cv2.COLOR_GRAY2RGB)
    
#     For all the blocks in the grid look for its best match in reference
    motion_vectors = []
    for h in range(nblocks_height):
        motion_vectors_h = []
        print("Nos movemos en vertical ",h)
        for w in range(nblocks_width):
            print(w)
            # Store block position
            y1 = h*m  # h/w refers to the block number
            y2 = y1+m  # y/x refers to the pixel in image (cartesian axes style)
            x1 = w*m
            x2 = x1+m
            block = frame_curr_gray[y1:y2, x1:x2]
            # Compare with surroundings (start with min mse as the one 
            #with same position)
            min_mse = mse_frames(block, frame_ref_gray[y1:y2, x1:x2])
            best_y_mov = 0
            best_x_mov = 0
#            # Check all surroundings areas in reference to get the one with smallest MSE, full search method
            for ref_y1 in range(max(y1-p, 0), min(y1+p, frames_height-m)+1):
                for ref_x1 in range(max(x1-p, 0), min(x1+p, frames_width-m)+1):
                    mse = mse_frames(block, frame_ref_gray[ref_y1:ref_y1+m, ref_x1:ref_x1+m])
                    if mse < min_mse:
                        min_mse = mse
                        best_y_mov = y1 - ref_y1
                        best_x_mov = x1 - ref_x1
            # Save motion vector
            motion_vectors_h += [[best_x_mov, best_y_mov]]
            if (best_x_mov != 0) | (best_y_mov != 0):
                print(h, w)
                print(best_y_mov, best_x_mov, min_mse)
                # Draw legend
                cv2.putText(copy, "Current frame", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(copy, "Reference frame", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                # Draw rectangle of the block and its best match en reference frame
                cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red (where it is)
                cv2.rectangle(copy, (x1-best_x_mov, y1-best_y_mov), (x2-best_x_mov, y2-best_y_mov), (255, 0, 0), 1) # Blue (where it was)
                # Draw motion vector from where it was to where it is now
                cv2.arrowedLine(copy, (int(x1+m/2), int(y1+m/2)), (int(x1+m/2-best_x_mov), int(y1+m/2-best_y_mov)), (255, 0,0), 1)
        motion_vectors += [motion_vectors_h]
    cv2.imwrite(vpath+str(n)+"out.png", copy)  
    cv2.imwrite(vpath+str(n-1)+"refer.png", frame_ref_gray)                         
    print("Block matching finished!")                

# Cleanup the camera and close any open windows
#camera.release()
#cv2.destroyAllWindows()
    