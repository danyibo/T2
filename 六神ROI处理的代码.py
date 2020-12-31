import os
import numpy as np
import matplotlib.pyplot as plt
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import get_array_from_path, standard
from Tool.FolderProcress import make_folder


def get_2d_roi():
    root_path = r"Y:\LS\liusheng _shidaoai_train_and_test"
    store_path = r"Y:\LS\xxxxxxxxxxxx"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        store_folder_path = make_folder(os.path.join(store_path, folder))
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            data_path = os.path.join(case_path, "data.nii.gz")
            roi_path = os.path.join(case_path, "ROI.nii.gz")
            data_array = get_array_from_path(data_path)
            roi_array = get_array_from_path(roi_path)
            # 存放一个方向的结果
            for i in range(roi_array.shape[-1]):
                if np.sum(roi_array[..., i]) != 0:
                    # 开始存放：
                    # data 1 是从一个方向看的结果
                    data_1 = data_array[..., i-1:i+2]
                    roi_1 = roi_array[..., i-1:i+2]
                    store_case_path = make_folder(os.path.join(store_folder_path, str(i)+"_1_"+case))
                    np.save(os.path.join(store_case_path, "data.npy"), data_1)
                    np.save(os.path.join(store_case_path, "roi.npy"), roi_1)
                    print("---1---")
            # 存放另一个方向的结果
            for j in range(roi_array.shape[0]):
                if np.sum(roi_array[j, ...]) != 0:
                    # 开始存放
                    # data 2 是从第二个方向看的结果
                    data_2 = data_array[j-1:j+2, ...]
                    roi_2 = roi_array[j-1:j+2, ...]
                    store_case_path = make_folder(os.path.join(store_folder_path, str(j) + "_2_" + case))
                    np.save(os.path.join(store_case_path, "data.npy"), data_2)
                    np.save(os.path.join(store_case_path, "roi.npy"), roi_2)
                    print("---2---")
            # 存放第三个方向看的结果
            for k in range(roi_array.shape[1]):
                if np.sum(roi_array[:, k, :]) != 0:
                    # 开始存放第三个方向的结果  data 3
                    data_3 = data_array[:, k-1:k+2, :]
                    roi_3 = roi_array[:, k-1:k+2, :]
                    store_case_path = make_folder(os.path.join(store_folder_path, str(k) + "_3_" + case))
                    np.save(os.path.join(store_case_path, "data.npy"), data_3)
                    np.save(os.path.join(store_case_path, "roi.npy"), roi_3)
                    print("---3---")
            print("case {} is finished!".format(case))

# get_2d_roi()

# 进行膨胀处理
import skimage
from skimage.morphology import binary_dilation


def get_result():
    data_path = r"Y:\DYB\shidaoai_vibenii\649302\BLADE42_256_3mm\026_BLADE42_256_3mm.nii"
    roi_path = r"Y:\DYB\shidaoai_vibenii\649302\BLADE42_256_3mm\026_BLADE42_256_3mm_ROI.mha"
    data_array = standard(get_array_from_path(data_path))
    roi_array = get_array_from_path(roi_path)
    new_roi = binary_dilation(roi_array)
    Imshow3DArray(data_array, [roi_array, new_roi*1])

get_result()