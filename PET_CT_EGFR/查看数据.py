"""再次查看一下数据的情况"""

import os
import numpy as np
import pandas as pd
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
import matplotlib.pyplot as plt
import SimpleITK as sitk
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef
from Tool.FolderProcress import make_folder
import shutil




def check_data():
    case_path = r"Y:\DYB\PETCT_EGFR\EGFR+\004+"
    ct_path = os.path.join(case_path, "ct.nii")
    pet_path = os.path.join(case_path, "pet_Resize.nii")
    roi_path = os.path.join(case_path, "ct_roi.nii")
    ct_array = get_array_from_path(ct_path)
    pet_array = get_array_from_path(pet_path)
    roi_array = get_array_from_path(roi_path)
    ct_array = standard(ct_array)
    pet_array = standard(pet_array)
    pet_array = np.flipud(pet_array)
    Imshow3DArray(ct_array, roi_array)


def flup_pet():
    """将pet的数据进行上下颠倒"""
    root_path = r"Y:\DYB\PETCT_EGFR"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            pet_path = os.path.join(case_path, "pet_Resize.nii")
            pet_array = get_array_from_path(pet_path)
            pet_array = np.flipud(pet_array)
            pet = sitk.ReadImage(pet_path)
            SaveArrayToNiiByRef(store_path=os.path.join(case_path, "new_pet_resize.nii"),
                                array=pet_array,
                                ref_image=pet)
            print("case {} is finished!".format(case))


def check_all_case_roi():
    root_path = r"Y:\DYB\PETCT_EGFR"
    store_path = r"Y:\DYB\2020832DATA\doctor_gao\EGFR_PET_CT_DATA"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        store_folder_path = make_folder(os.path.join(store_path, folder))
        for case in os.listdir(folder_path):
            store_case_path = make_folder(os.path.join(store_folder_path, case))
            case_path = os.path.join(folder_path, case)
            ct_path = os.path.join(case_path, "ct.nii")
            pet_path = os.path.join(case_path, "new_pet_resize.nii")
            roi_path = os.path.join(case_path, "ct_roi.nii")
            shutil.copy(ct_path, store_case_path)
            shutil.copy(pet_path, store_case_path)
            shutil.copy(roi_path, store_case_path)
            print("case {} is finished!".format(case))


            # roi_array = get_array_from_path(roi_path)
            # ct_array = get_array_from_path(ct_path)
            # pet_array = get_array_from_path(pet_path)
            # ct_and_pet_array = ct_array + pet_array
            # ct_and_pet_array = standard(ct_and_pet_array)
            # Imshow3DArray(ct_and_pet_array, roi_array)
            # for i in range(ct_and_pet_array.shape[-1]):
            #     if np.sum(roi_array[..., i]) != 0:
            #         plt.imshow(ct_and_pet_array[..., i], cmap="gray")
            #         plt.contour(roi_array[..., i])
            #         plt.show()
            #         break

import shutil
import random

def show_roi():
    """检查ROI因为非常慢，所以先存下来"""
    root_path = r"Y:\DYB\2020832DATA\doctor_gao\EGFR_PET_CT_DATA"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        # while True:
        #     random.shuffle(os.listdir(folder_path))  # 这个函数是没有返回值的
        for case in os.listdir(folder_path):
            # ct_path = os.path.join(folder_path, case, "ct.nii")
            # roi_path = os.path.join(folder_path, case, "roi.nii")
            # pet_path = os.path.join(folder_path, case, "pet.nii")
            # cp_path = os.path.join(folder_path, case, "cp.nii")
            cp_path = os.path.join(folder_path, case, "cp.nii.gz")
            try:

                os.remove(cp_path)
            except: pass
            print(case)
            # for i in range(ct_array.shape[-1]):
            #     if np.sum(roi_array[..., i]) != 0:
            #         plt.imshow(ct_and_pet_array[..., i], cmap="gray")
            #         plt.contour(roi_array[..., i])
            #         plt.show()




""""检查croped后的数据结果"""

# def check_croped_data():
#     root_path = r"Y:\DYB\PETCT_EGFR"
#     for folder in os.listdir(root_path):
#         folder_path = os.path.join(root_path, "EGFR+")
#         # for case in os.listdir(folder_path):
#         case_path = os.path.join(folder_path, "029+")
#         ct_path = os.path.join(case_path, "result_ct.nii")
#         roi_path = os.path.join(case_path, "result_roi.nii")
#         ct_array = get_array_from_path(ct_path)
#         roi_array = get_array_from_path(roi_path)
#         ct_array = np.transpose(ct_array, (2, 0, 1))
#         roi_array = np.transpose(roi_array, (2, 0, 1))
#         for i in range(ct_array.shape[-1]):
#             if np.sum(roi_array[..., i]) != 0:
#                 plt.imshow(ct_array[..., i], cmap="gray")
#                 plt.contour(roi_array[..., i])
#                 plt.show()




if __name__ == '__main__':
    # 检查一个case的数据，发现pet的数据需要颠倒一下
    # check_data()
    # flup_pet()
    # check_all_case_roi()
    show_roi()
    # check_croped_data()