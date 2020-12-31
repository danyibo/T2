import os
import cv2
import numpy as np
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef

###################
#  分割前的预处理
###################



def get_roi_index(roi_all):
    roi_sum = np.sum(roi_all, axis=(0, 1))
    roi_index = list(np.where(roi_sum != 0)[0])
    if roi_index[0] != 0:
        roi_index.insert(0, roi_index[0] - 1)
    if roi_index[-1] != roi_sum.shape[0] - 1:
        roi_index.append(roi_index[-1] + 1)
    return roi_index

def get_crop_shape(roi_all):

    h_roi = np.sum(roi_all, axis=0)  # shape(height, channels)
    h_roi = np.sum(h_roi, axis=1)  # 将数据加和到高这个维度上，shape(heights,)
    h_corp = np.where(h_roi != 0)[0]
    h_left, h_right = np.min(h_corp) - 1, np.max(h_corp) + 2

    w_roi = np.sum(roi_all, axis=1)
    w_roi = np.sum(w_roi, axis=1)
    w_crop = np.where(w_roi != 0)[0]
    w_top, w_bottom = np.min(w_crop) - 1, np.max(w_crop) + 2
    return h_left, h_right, w_top, w_bottom


def crop_data_array(roi_all, data_array):
    h_left, h_right, w_top, w_bottom = get_crop_shape(roi_all=roi_all)
    index_roi = get_roi_index(roi_all=roi_all)
    crop_array = data_array[w_top - 15: w_bottom + 15, h_left - 15: h_right + 15, :]
    return crop_array


def get_croped_data_roi(data_path, roi_path, store_path):
    data_array = get_array_from_path(data_path)
    # data_array = np.flipud(data_array)  # 注意这里的数据要进行翻转的
    roi_array = get_array_from_path(roi_path)
    data_array = standard(data_array)

    crop_array = crop_data_array(roi_array, data_array)
    def resize_data(data_array, resized_shape):
        resized_data = cv2.resize(data_array, resized_shape, interpolation=cv2.INTER_NEAREST)
        return resized_data

    roi_new = crop_data_array(roi_array, roi_array)
    new_array = resize_data(crop_array, (224, 224))
    roi_new = resize_data(roi_new, (224, 224))

    # Imshow3DArray(standard(new_array), roi_new)

    np.save(os.path.join(store_path, "croped_pet.npy"), new_array)
    np.save(os.path.join(store_path, "croped_roi.npy"), roi_new)


if __name__ == '__main__':
    root_path = r"Y:\DYB\2020832DATA\doctor_gao\PET_EGFR\EGFR_PET_CT_DATA"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            data_path = os.path.join(case_path, "pet.nii.gz")  # CT图像的路径
            roi_path = os.path.join(case_path, "roi.nii.gz")  # roi的路径


            get_croped_data_roi(data_path=data_path, roi_path=roi_path, store_path=case_path)
            print("Case {} is finished!".format(case))