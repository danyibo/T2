import os
import numpy as np
from skimage import filters
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
from skimage.measure import label
import skimage
from skimage import measure
import matplotlib.pyplot as plt
from MeDIT.ArrayProcess import SmoothRoi, RemoveSmallRegion, ExtractBoundaryOfRoi
from skimage.morphology import remove_small_objects

import cv2

def get_image():
    data_path = r"Y:\DYB\chang\BAO CAI ZHEN_2769351_sag_Resample.nii"
    data_array = get_array_from_path(data_path)
    data_array = standard(data_array)
    th = filters.threshold_otsu(data_array)
    roi_array = np.where(data_array > th/6.1, 1, 0)
    roi_array = cv2.blur(roi_array, (12, 12))
    Imshow3DArray(data_array, roi_array)

get_image()
