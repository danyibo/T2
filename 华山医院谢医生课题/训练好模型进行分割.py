import os
import numpy as np
# import keras

import matplotlib.pyplot as plt
from Tool.DataProcress import get_array_from_path, standard
from Tool.FolderProcress import make_folder
import shutil
from MeDIT.Visualization import Imshow3DArray
import SimpleITK as sitk
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef


def get_big_roi(roi_array):
    roi_sum = list(np.sum(roi_array, axis=(0, 1)))
    big_roi_index = roi_sum.index(max(roi_sum))
    new_roi = np.zeros(roi_array.shape)
    for i in range(roi_array.shape[-1]):
        if i != big_roi_index:
            new_roi[..., i] = new_roi[..., i]
        else:
            new_roi[..., i] = roi_array[..., i]
    return new_roi


def get_roi():
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\Normal_data_nii_and_T2"
    for case in os.listdir(root_path)[4:]:
        case_path = os.path.join(root_path, case)
        data_path = os.path.join(case_path, "data.nii")
        data = sitk.ReadImage(data_path)
        data_array = get_array_from_path(data_path)
        data_array = standard(data_array)
        data_array = np.flipud(data_array)
        roi_name = ["roi_" + str(i) + ".npy"for i in range(1, 5)]
        roi_path = [os.path.join(case_path, i) for i in roi_name]
        roi_array_list = [np.load(i) for i in roi_path]
        # roi_array_list = [get_big_roi(i) for i in roi_array_list]

        roi_1 = np.where(roi_array_list[0] == 1, 1, 0)
        roi_2 = np.where(roi_array_list[1] == 1, 2, 0)
        roi_3 = np.where(roi_array_list[2] == 1, 3, 0)
        roi_4 = np.where(roi_array_list[3] == 1, 4, 0)
        # roi_1 = get_big_roi(roi_1)
        # roi_2 = get_big_roi(roi_2)
        # roi_3 = get_big_roi(roi_3)
        # roi_4 = get_big_roi(roi_4)

        # big_roi = roi_1 + roi_2 + roi_3 + roi_4
        # print(np.sum(roi_1), np.sum(roi_2), np.sum(roi_3), np.sum(roi_4))
        Imshow3DArray(data_array, roi_array_list)

        # SaveArrayToNiiByRef(store_path = os.path.join(case_path, "roi.nii.gz"),
        #                     array=big_roi,
        #                     ref_image=data)
        print(case)



def check_roi():
    """检查之前ROI中的数值"""
    case_path = r"Y:\DYB\data_and_result\doctor_xie\knee_data\CAI NAI JI-20140605"
    data_path = os.path.join(case_path, "data_1.nii")
    roi_path = os.path.join(case_path, "roi.nii.gz")
    data_array = standard(get_array_from_path(data_path))
    data_array = np.flipud(data_array)
    roi_array = get_array_from_path(roi_path)
    roi_1 = np.where(roi_array == 1, 1, 0)  # 上软骨后面的层
    roi_2 = np.where(roi_array==2, 1, 0)  # 上软骨前面的层
    roi_3 = np.where(roi_array==3, 1, 0)  # 下软骨后面的层
    roi_4 = np.where(roi_array==4, 1, 0)
    roi_5 = np.where(roi_array==5, 1, 0)
    roi_6 = np.where(roi_array==6, 1, 0)
    Imshow3DArray(standard(data_array), [roi_1, roi_2, roi_3, roi_4])




def get_dicom_and_roi():
    """将数据的dicom和roi放在一起"""
    root_dicom_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    root_roi_path = r"Y:\DYB\2020832DATA\doctor_xie\Normal_data_nii_and_T2"
    store_path = r"Y:\DYB\2020832DATA\doctor_xie\data"
    for dicom, roi in zip(os.listdir(root_dicom_path), os.listdir(root_roi_path)):
        dicom_path = os.path.join(root_dicom_path, dicom, "folder_1")
        roi_path = os.path.join(root_roi_path, roi, "roi.nii")
        store_case_path = make_folder(os.path.join(store_path, dicom))
        store_case_dicom = make_folder(os.path.join(store_case_path, "dicom"))
        shutil.copy(roi_path, store_case_path)
        for file in os.listdir(dicom_path):
            dicom_file_path = os.path.join(os.path.join(dicom_path, file))
            shutil.copy(dicom_file_path, store_case_dicom)
        print(dicom)

# get_dicom_and_roi()

def get_new_roi():
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\data"
    for case in os.listdir(root_path):
        roi_path = os.path.join(root_path, case, "new_roi.nii")
        try:
            os.remove(roi_path)
        except: pass

        # roi = sitk.ReadImage(roi_path)
        # roi_array = get_array_from_path(roi_path)
        # roi_array = np.flipud(roi_array)
        # SaveArrayToNiiByRef(os.path.join(root_path, case, "roi.nii"),
        #                     roi_array,
        #                     roi)
        # print(case)
get_new_roi()





# get_roi()
# check_roi()