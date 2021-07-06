import datetime
#import imutils
import cv2
import numpy as np
import pandas as pd
#from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import time
from os import listdir
from os.path import isfile, join
import time


def mse_frames(frame1, frame2):
    df_error = pd.DataFrame({'original':np.array(frame1.flatten(), dtype = 'f'), 'transformed':np.array(frame2.flatten(), dtype = 'f')})
    df_error['error'] = (df_error['original'] - df_error['transformed'])**2
    return sum(df_error['error']) / len(df_error)

def mae_frames(frame1, frame2):
    df_error = pd.DataFrame({'original':np.array(frame1.flatten(), dtype = 'f'), 'transformed':np.array(frame2.flatten(), dtype = 'f')})
    df_error['error'] = (df_error['original'] - df_error['transformed']).abs()
    return sum(df_error['error']) / len(df_error)

# Read the video
vname="mono.mp4"
vpath="C:\\Users\\Borja042\\Desktop\\MASTER2\\COMPRESION\\LABS567\\OUT6"
video=vname
print(video)

print(video)
camera = cv2.VideoCapture(video)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frames_width = int(camera.get(3))
frames_height = int(camera.get(4))

print("Video width in px:", frames_width)
print("Video height in px:", frames_height)

m = 16
p = 3
nblocks_width = int(frames_width / m)
nblocks_height = int(frames_height / m)

print("Block size:", m)
print("Pixels for search around:", p)

print("Number of horizontal blocks:", nblocks_width)
print("Number of vertical blocks:", nblocks_height)


# Loop over the frames of the video until we get the one we want
i = 1
n = 49
while i<=n:
    
    # Grab another frame
    (grabbed, frame_curr) = camera.read()
 
#    print("Frame: "+str(i))   

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
    
    # For all the blocks in the grid look for its best match in reference
    motion_vectors = []
    for h in range(nblocks_height):
        motion_vectors_h = []
        for w in range(nblocks_width):
            # Store block position
            y1 = h*m  # h/w refers to the block number
            y2 = y1+m  # y/x refers to the pixel in image (cartesian axes style)
            x1 = w*m
            x2 = x1+m
            block = frame_curr_gray[y1:y2, x1:x2]
            # Compare with surroundings (start with min mse as the one with same position)
            min_mse = mse_frames(block, frame_ref_gray[y1:y2, x1:x2])
            best_y_mov = 0
            best_x_mov = 0
            # Check all surroundings areas in reference to get the one with smallest MSE, full search method
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
camera.release()
cv2.destroyAllWindows()

frame_motion_comp = frame_ref_gray.copy()
# For all the blocks in the grid reposition them based on the motion vectors
for h in range(nblocks_height):
    for w in range(nblocks_width):
#        print(" ")
        print("Estamos en y=",h ,"x=", w)
        # Store block position
        y1 = h*m  # h/w refers to the block number
        y2 = y1+m  # y/x refers to the pixel in image (cartesian axes style)
        x1 = w*m
        x2 = x1+m
        mv = motion_vectors[h][w]
        print("Bloque se mueve", motion_vectors[h][w])
        y_mov = mv[1]
        x_mov = mv[0]
#        print(frame_motion_comp)
#        print('y1,y2',y1,y2,'x1,x2',x1,x2)
        print("movimiento", mv)
#        print(frame_ref_gray[y1-y_mov:y2-y_mov, x1-x_mov:x2-x_mov])
#        plt.imshow(frame_motion_comp[y1:y2, x1:x2], cmap = 'gray')
#        plt.show()
        frame_motion_comp[y1:y2, x1:x2] = frame_ref_gray[y1-y_mov:y2-y_mov, x1-x_mov:x2-x_mov]
#        plt.imshow(frame_motion_comp[y1:y2, x1:x2], cmap = 'gray')
#        plt.show()

plt.imshow(frame_ref_gray, cmap = 'gray')
plt.title("Reference Frame")
plt.show()
plt.imshow(frame_curr_gray, cmap = 'gray')
plt.title("Current Frame")
plt.show()
#cv2.imwrite(vpath+str(n)+"motioncomp.png", frame_motion_comp)
plt.imshow(frame_motion_comp, cmap = 'gray')
plt.title("Motion Compensated Frame")
plt.show()


dif = frame_curr_gray - frame_ref_gray
dif1 = sum(sum(dif))
eres = frame_curr_gray - frame_motion_comp
dif2 = sum(sum(eres))
extra = frame_ref_gray - frame_motion_comp
dif3 = sum(sum(extra))
#
plt.imshow(dif, cmap = 'gray')
plt.title("Fc - Fr")
plt.show()
plt.imshow(eres, cmap = 'gray')
plt.title("Fc - Fcc")
plt.show()
plt.imshow(extra, cmap = 'gray')
plt.title("Fr - Fcc")
plt.show()
orig_frames_dif = mae_frames(frame_curr_gray, frame_ref_gray)
print("Difference between current and reference:", orig_frames_dif)
motion_compensated_error = mae_frames(frame_curr_gray, frame_motion_comp)
print("Difference between current and predicted (Motion compensated error):", motion_compensated_error)
ref_motion_compensated_error = mae_frames(frame_ref_gray, frame_motion_comp)
print("Difference between current and predicted (Motion compensated error):", ref_motion_compensated_error)



# PERFORM ON SEVERAL PAIRS
camera = cv2.VideoCapture(video)

# Grab first frame
(grabbed, frame_curr) = camera.read()
frame_ref_gray = cv2.cvtColor(frame_curr.copy(), cv2.COLOR_BGR2GRAY)

m = 4
p = 3
start_time = time.time()
# Loop over the frames of the video
i = 50
n = 60
orig_frames_dif_list = []
motion_compensated_error_list = []

while i<n:
    
    # Grab another frame
    (grabbed, frame_curr) = camera.read()
    i=i+1     

    print("Frame: "+str(i))   

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
            y1 = h*m  # h/w refers to the block number
            y2 = y1+m  # y/x refers to the pixel in image (cartesian axes style)
            x1 = w*m
            x2 = x1+m
            block = frame_curr_gray[y1:y2, x1:x2]
            # Compare with surroundings (start with min mse as the one with same position)
            min_mse = mse_frames(block, frame_ref_gray[y1:y2, x1:x2])
            best_y_mov = 0
            best_x_mov = 0
            # Check all surroundings areas in reference to get the one with smallest MSE, full search method
            for ref_y1 in range(max(y1-p, 0), min(y1+p, frames_height-m)+1):
                for ref_x1 in range(max(x1-p, 0), min(x1+p, frames_width-m)+1):
                    mse = mse_frames(block, frame_ref_gray[ref_y1:ref_y1+m, ref_x1:ref_x1+m])
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
            y1 = h*m  # h/w refers to the block number
            y2 = y1+m  # y/x refers to the pixel in image (cartesian axes style)
            x1 = w*m
            x2 = x1+m
            mv = motion_vectors[h][w]
            #print(h, w, mv, x1, x2, y1, )
            y_mov = mv[1]
            x_mov = mv[0]
            frame_motion_comp[y1:y2, x1:x2] = frame_ref_gray[y1-y_mov:y2-y_mov, x1-x_mov:x2-x_mov]

    # Compute the movement measure and the motion compensated error
    orig_frames_dif = mae_frames(frame_curr_gray, frame_ref_gray)
    orig_frames_dif_list += [orig_frames_dif]
    print("Difference original:", orig_frames_dif)
    motion_compensated_error = mae_frames(frame_curr_gray, frame_motion_comp)
    motion_compensated_error_list += [motion_compensated_error]
    print("Motion compensated error:", motion_compensated_error)
    print("__________________________")
    
    #Keep curr for next
    frame_ref_gray = cv2.cvtColor(frame_curr.copy(), cv2.COLOR_BGR2GRAY)    
elapsed_time = time.time() - start_time
print ("Processing time with p = ",p,"time", elapsed_time)
# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()