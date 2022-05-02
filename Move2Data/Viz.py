# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import ConnectionPatch
import cv2
import numpy as np
import glob
import subprocess

# %%
def get_plots(ob, elev=270, azim=90):
    '''
    This function plots the human position from the 3D coordinates.

    Args:
        ob: Single observation from dataframe. This, should include the following key points: nose, left_eye, 
            right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow,
            left_knee, right_knee, left_anke, right_ankle and each of them should be repeated 3 times having the
            x,y and z coordinates in this format: x_nose, y_nose, z_nose
        elev: Elevation angle in the vertical plane in degrees.
        azim: Azimuth angle in the horizontal plane in degrees

    Returns:
        This function returns a plot of the human position in that observation.
    '''
    #get points
    x_nle, y_nle, z_nle = [ob.x_nose, ob.x_left_eye], [ob.y_nose,ob.y_left_eye], [ob.z_nose, ob.z_left_eye]
    x_lelea, y_lelea, z_lelea = [ob.x_left_ear, ob.x_left_eye], [ob.y_left_ear,ob.y_left_eye], [ob.z_left_ear, ob.z_left_eye]
    x_nre, y_nre, z_nre = [ob.x_nose, ob.x_right_eye], [ob.y_nose,ob.y_right_eye], [ob.z_nose, ob.z_right_eye]
    x_rerea, y_rerea, z_rerea = [ob.x_right_ear, ob.x_right_eye], [ob.y_right_ear,ob.y_right_eye], [ob.z_right_ear, ob.z_right_eye]
    x_rsls, y_rsls, z_rsls = [ob.x_right_shoulder, ob.x_left_shoulder], [ob.y_right_shoulder,ob.y_left_shoulder], [ob.z_right_shoulder, ob.z_left_shoulder]
    x_rsrh, y_rsrh, z_rsrh = [ob.x_right_shoulder, ob.x_right_hip], [ob.y_right_shoulder,ob.y_right_hip], [ob.z_right_shoulder, ob.z_right_hip]
    x_rsrel, y_rsrel, z_rsrel = [ob.x_right_shoulder, ob.x_right_elbow], [ob.y_right_shoulder,ob.y_right_elbow], [ob.z_right_shoulder, ob.z_right_elbow]
    x_rwrel, y_rwrel, z_rwrel = [ob.x_right_wrist, ob.x_right_elbow], [ob.y_right_wrist,ob.y_right_elbow], [ob.z_right_wrist, ob.z_right_elbow]
    x_lells, y_lells, z_lells = [ob.x_left_elbow, ob.x_left_shoulder], [ob.y_left_elbow,ob.y_left_shoulder], [ob.z_left_elbow, ob.z_left_shoulder]
    x_lhls, y_lhls, z_lhls = [ob.x_left_hip, ob.x_left_shoulder], [ob.y_left_hip,ob.y_left_shoulder], [ob.z_left_hip, ob.z_left_shoulder]
    x_lellw, y_lellw, z_lellw = [ob.x_left_elbow, ob.x_left_wrist], [ob.y_left_elbow,ob.y_left_wrist], [ob.z_left_elbow, ob.z_left_wrist]
    x_rkrh, y_rkrh, z_rkrh = [ob.x_right_knee, ob.x_right_hip], [ob.y_right_knee,ob.y_right_hip], [ob.z_right_knee, ob.z_right_hip]
    x_rkra, y_rkra, z_rkra = [ob.x_right_knee, ob.x_right_ankle], [ob.y_right_knee,ob.y_right_ankle], [ob.z_right_knee, ob.z_right_ankle]
    x_lhrh, y_lhrh, z_lhrh = [ob.x_left_hip, ob.x_right_hip], [ob.y_left_hip,ob.y_right_hip], [ob.z_left_hip, ob.z_right_hip]
    x_lhlk, y_lhlk, z_lhlk = [ob.x_left_hip, ob.x_left_knee], [ob.y_left_hip,ob.y_left_knee], [ob.z_left_hip, ob.z_left_knee]
    x_lalk, y_lalk, z_lalk = [ob.x_left_ankle, ob.x_left_knee], [ob.y_left_ankle,ob.y_left_knee], [ob.z_left_ankle, ob.z_left_knee]
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x_nle, y_nle, z_nle, c='red', s=20)
    ax.scatter(x_lelea, y_lelea, z_lelea, c='red', s=20)
    ax.scatter(x_nre, y_nre, z_nre, c='red', s=20)
    ax.scatter(x_rerea, y_rerea, z_rerea, c='red', s=20)
    ax.scatter(x_rsls, y_rsls, z_rsls, c='red', s=20)
    ax.scatter(x_rsrh, y_rsrh, z_rsrh, c='red', s=20)
    ax.scatter(x_rsrel, y_rsrel, z_rsrel, c='red', s=20)
    ax.scatter(x_rwrel, y_rwrel, z_rwrel, c='red', s=20)
    ax.scatter(x_lells, y_lells, z_lells, c='red', s=20)
    ax.scatter(x_lhls, y_lhls, z_lhls, c='red', s=20)
    ax.scatter(x_lellw, y_lellw, z_lellw, c='red', s=20)
    ax.scatter(x_rkrh, y_rkrh, z_rkrh, c='red', s=20)
    ax.scatter(x_rkra, y_rkra, z_rkra, c='red', s=20)
    ax.scatter(x_lhlk, y_lhlk, z_lhlk, c='red', s=20)
    ax.scatter(x_lhrh, y_lhrh, z_lhrh, c='red', s=20)
    ax.scatter(x_lalk, y_lalk, z_lalk, c='red', s=20)


    ax.plot(x_nle, y_nle, z_nle, color='black')
    ax.plot(x_lelea, y_lelea, z_lelea, color='black')
    ax.plot(x_nre, y_nre, z_nre, color='black')
    ax.plot(x_rerea, y_rerea, z_rerea,color='black')
    ax.plot(x_rsls, y_rsls, z_rsls,color='black')
    ax.plot(x_rsrh, y_rsrh, z_rsrh,color='black')
    ax.plot(x_rsrel, y_rsrel, z_rsrel,color='black')
    ax.plot(x_rwrel, y_rwrel, z_rwrel,color='black')
    ax.plot(x_lells, y_lells, z_lells,color='black')
    ax.plot(x_lhls, y_lhls, z_lhls,color='black')
    ax.plot(x_lellw, y_lellw, z_lellw,color='black')
    ax.plot(x_rkrh, y_rkrh, z_rkrh,color='black')
    ax.plot(x_rkra, y_rkra, z_rkra,color='black')
    ax.plot(x_lhlk, y_lhlk, z_lhlk,color='black')
    ax.plot(x_lhrh, y_lhrh, z_lhrh,color='black')
    ax.plot(x_lalk, y_lalk, z_lalk,color='black')

    ax.grid(False)
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)
    return ax


def get_avi(num_frames,path,avi_path, fps):
    '''
    This function creates an avi from the frames saved in a folder in format Video_0, Video_1 etc.

    Args:
        num_frames: number of frames you want to combine.
        path: path where you have stored the frames
        avi_path: path where you want to store the avi file
        fps: frames per second
    Returns:
        This function returns the avi file with all frames one after the other.
    '''
    img_array = []
    for frame in range(num_frames):
        filename= path+"Video_"+str(frame)+".png"
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(avi_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return True

def get_mp4(inputfile,outputfile):
    '''
    This function converts an avi file into a mp4 file.

    Args:
        inputfile: path to avi file.
        outputfile: path to mp4 file.
    Returns:
        This function returns an mp4 file.
    '''
    subprocess.call(['ffmpeg', '-i', inputfile, outputfile])  
    return True
