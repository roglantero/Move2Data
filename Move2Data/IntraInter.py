# %%
import pandas as pd
import numpy as np

# %%
def intra_inter_transform(keypoints,df, reference_joint):
    '''
    This function performs the intra- and inter-movement transformation.

    Args:
    keypoints: List of keypoints that are being used, and from which you have the 3D coordinates, excluding the reference joint.
    df: Pandas dataframe with all 3D coordinates of the keypoints in the format: x_nose, y_nose, z_nose, including the reference joint.
    reference_joint: reference joint name.

    Returns:
    This function returns a pandas Dataframe with the transformed data.
    '''
    dfq= pd.DataFrame([])
    for keypoint in keypoints:
        x= df["x_"+reference_joint]-df["x_"+keypoint]
        y= df["y_"+reference_joint]-df["y_"+keypoint]
        dfq["r_"+keypoint]= np.sqrt(x**2+y**2)
        dfq["sin_"+keypoint]= y / dfq["r_"+keypoint]
        dfq["cos_"+keypoint]= x/ dfq["r_"+keypoint]
        dfq["z_"+keypoint]= df["z_"+keypoint]
    dfq['x_'+reference_joint] = df['x_'+reference_joint]
    dfq['y_'+reference_joint] = df['y_'+reference_joint]
    dfq['z_'+reference_joint] = df['z_'+reference_joint] 
    return dfq

def intra_backtransform(keypoints, dfq, reference_joint):
    '''
    This function backtransforms the intra- and inter-movement transformation.

    Args:
    keypoints: List of keypoints that are being used, and from which you have the 3D coordinates, excluding the reference joint.
    dfq: Pandas dataframe with inter- and intra-movement transformation.
    reference_joint: reference joint name.

    Returns:
    This function returns a pandas Dataframe with the transformed data.
    '''
    bk= pd.DataFrame([])
    for keypoint in keypoints:
        x= dfq["cos_"+keypoint]*dfq["r_"+keypoint]
        bk["x_"+keypoint]= dfq["x_left_hip"] + x
        y= dfq["sin_"+keypoint]*dfq["r_"+keypoint]
        bk["y_"+keypoint]= dfq["y_left_hip"] + y
        bk["z_"+keypoint]= dfq["z_"+keypoint]
    bk['x_'+reference_joint] = dfq['x_'+reference_joint]
    bk['y_'+reference_joint] = dfq['y_'+reference_joint]
    bk['z_'+reference_joint] = dfq['z_'+reference_joint]
    return bk


