import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
from Tool.FolderProcress import make_folder


def check_data_roi():
    """检查数据和ROI"""
    root_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data"
    store_path = r"Y:\DYB\data_and_result\doctor_xie\knn_segment_all_roi"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        store_case_path = make_folder(os.path.join(store_path, case))
        # 选择分割的数据是的一个回波的数据
        data_path = os.path.join(case_path, "data_1.nii")
        roi_path = os.path.join(case_path, "roi.nii.gz")
        data_array = standard(get_array_from_path(data_path))
        data_array = np.flipud(data_array)
        roi_array = get_array_from_path(roi_path)
        roi_array_list = [np.where(roi_array == i, 1, 0) for i in np.unique(roi_array)[1:]]

        np.save(os.path.join(store_case_path, "data.npy"), data_array)
        np.save(os.path.join(store_case_path, "all_roi.npy"), roi_array_list)
        print(case, " is finished!")


def crop_data_roi():
    """将数据裁剪一下"""
    store_path = r"Y:\DYB\data_and_result\doctor_xie\knn_segment_all_roi"
    for case in os.listdir(store_path):
        case_path = os.path.join(store_path, case)
        data_path = os.path.join(case_path, "data.npy")
        roi_path = os.path.join(case_path, "all_roi.npy")
        data_array = np.load(data_path)
        roi_array = np.load(roi_path)
        # Imshow3DArray(standard(data_array), list(roi_array))

def check_normal_data():
    """检查对照组的数据大小是否和训练组的相同"""
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case, "nii_folder")
        for file in os.listdir(case_path):
            file_path = os.path.join(case_path, file)
            file_array = get_array_from_path(file_path)
            print(file_array.shape)
    # 检查后是相同的，因此可以直接处理



# check_data_roi()
# crop_data_roi()
# check_normal_data()