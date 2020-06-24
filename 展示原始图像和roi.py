import os
import numpy as np
from MeDIT.Visualization import Imshow3DArray
import SimpleITK as sitk
from Ankle.tool import DataProcess

def show_data_roi(case_path):
    data_path = os.path.join(case_path, "data_1.nii")
    roi_path = os.path.join(case_path, "roi.nii")
    data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
    roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
    roi_array = np.transpose(roi, (1, 2, 0))
    roi_array_1 = np.where(roi_array == 1, 1, 0)
    roi_array_2 = np.where(roi_array == 2, 1, 0)
    roi_array_3 = np.where(roi_array == 3, 1, 0)
    roi_array_4 = np.where(roi_array == 4, 1, 0)
    roi_array_5 = np.where(roi_array == 5, 1, 0)
    roi_array_6 = np.where(roi_array == 6, 1, 0)
    roi_list = [roi_array_1, roi_array_2, roi_array_3, roi_array_4, roi_array_5, roi_array_6]
    data = np.transpose(data, (1, 2, 0))
    data = np.flipud(data)
    data = DataProcess.standard(data)
    Imshow3DArray(data, roi_list)

all_data_path = r'E:\split_data_ACLR_T2'
for case in os.listdir(all_data_path):
    case_path = os.path.join(all_data_path, case)
    show_data_roi(case_path)
