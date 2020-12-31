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
import shutil
from Tool.FolderProcress import make_folder


def check_data_roi():
    """检查数值问题"""
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\健康T2勾画20201229\健康T2勾画20201229"
    store_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    for case in os.listdir(root_path):
        roi_path = os.path.join(root_path, case, "roi.nii")
        shutil.copy(roi_path, os.path.join(store_path, case, "roi.nii"))
        print(case)


check_data_roi()

