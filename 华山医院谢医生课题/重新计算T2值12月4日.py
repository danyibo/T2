import os
import SimpleITK as sitk
import pydicom
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

from Tool.DataProcress import get_array_from_path
from Tool.FolderProcress import make_folder
from ALL.MIP4AIM.MIP4AIM.Dicom2Nii.Dicom2Nii import ProcessSeries
from Tool.Visualization import show_array

# plt.style.use("ggplot")

"""

  计算T2的思路
  （1）先把所有的dicom数据按照回波的拆成五个文件夹，分别存放每个回波的图像
  （2）把每个回波的图像合成nii图像
  （3）TE从dicom文件夹中读取即可
  （4）随着1-5，TE逐渐增大，信号强度逐渐减小
  2020年12月5日 淡一波

"""


def get_TE(dicom_file_path):
    image = sitk.ReadImage(dicom_file_path)
    image_array = sitk.GetArrayFromImage(image)
    image_shape = image_array.shape
    image_array = image_array.flatten()
    image_info = pydicom.dcmread(dicom_file_path)
    TE = float(image_info.EchoTime)
    return image_array, TE, image_shape


def sort_file(dicom_folder_path):
    """对dicom的文件进行排序"""
    list_new_image = []
    list_number = []
    image_list = os.listdir(dicom_folder_path)
    for i in image_list:
        if i.split(".")[-1] != "npy":
            list_number.append(int(i.split("M")[-1]))

        list_number.sort()
    for j in list_number:

        for k in os.listdir(dicom_folder_path):
            if k.split(".")[-1] != "npy":
                if j == int(k.split("M")[-1]):
                    list_new_image.append(k)
    return list_new_image


def get_TE_in_one_case(case_path):
    """获取一个Case的TE"""
    dicom_folder_path = os.path.join(case_path, "all_dicom")
    TE_list = []
    for file in sort_file(dicom_folder_path):
        dicom_file_path = os.path.join(dicom_folder_path, file)
        _, TE, _ = get_TE(dicom_file_path)
        TE_list.append(TE)
    return sorted(list(set(TE_list)))


def get_dicom_by_echo():
    """获取dicom按照回波的顺序进行排序保存"""
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)

        # 创建保存五个回波的文件夹
        save_folder_name = ["new_folder_" + str(i) for i in range(1, 6)]
        save_folder_path = [os.path.join(case_path, i) for i in save_folder_name]

        # case TE 是一个case的TE值
        case_TE = get_TE_in_one_case(case_path)

        dicom_folder_path = os.path.join(case_path, "all_dicom")
        for file in sort_file(dicom_folder_path):  # 务必进行排序
            dicom_path = os.path.join(dicom_folder_path, file)  # 每一张dicom的路径
            _, TE, _ = get_TE(dicom_path)
            if TE == case_TE[0]:
                shutil.copy(dicom_path, make_folder(save_folder_path[0]))
            elif TE == case_TE[1]:
                shutil.copy(dicom_path, make_folder(save_folder_path[1]))
            elif TE == case_TE[2]:
                shutil.copy(dicom_path, make_folder(save_folder_path[2]))
            elif TE == case_TE[3]:
                shutil.copy(dicom_path, make_folder(save_folder_path[3]))
            else:
                shutil.copy(dicom_path, make_folder(save_folder_path[4]))
        print("case {} is finished!".format(case))


def get_nii_by_echo():
    """将每个回波的图像转为nii"""
    # 正常对照组
    # root_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    # 有病的组
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\split_data_ACLR_T2"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)

        # 创建保存五个回波的文件夹
        save_folder_name = ["folder_" + str(i) for i in range(1, 6)]
        save_folder_path = [os.path.join(case_path, i) for i in save_folder_name]
        for folder in save_folder_path:
            save_folder = make_folder(os.path.join(case_path, folder+"_nii"))
            folder_path = os.path.join(case_path, folder)
            ProcessSeries(folder_path, save_folder)


def get_one_folder_TE(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        _, TE, _ = get_TE(file_path)
        return TE


def get_nii_array(folder_path):
    for file in os.listdir(folder_path):
        nii_path = os.path.join(folder_path, file)
        nii_array = np.flipud(get_array_from_path(nii_path))
        nii_array = nii_array.flatten()
        return nii_array
        # return nii_array


def get_data_shape(folder_path):
    for file in os.listdir(folder_path):
        nii_path = os.path.join(folder_path, file)
        nii_array = get_array_from_path(nii_path)
        return nii_array.shape


def compute_t2_by_four_points(p_2, p_3, p_4, p_5, TE_list):
    TE = np.array(TE_list).reshape(-1, 1)
    ln_S_list = []  # 信号强度取对数
    S_list = [p_2, p_3, p_4, p_5]  # 原始信号强度
    for i in S_list:
        if i == 0:
            ln_S_list.append(i)   # 为零，则取对数也设置为零
        else:
            ln_S_list.append(math.log(i))
    model = LinearRegression()  # 拟合直线模型
    model.fit(TE, ln_S_list)
    # y_plot = model.predict(TE)
    # plt.scatter(TE, ln_S_list, color='red', label="point", linewidth=7, alpha=0.8)
    # plt.plot(TE, y_plot, color="green", label="line", linewidth=4, alpha=0.8)
    # plt.xlabel("TE")
    # plt.ylabel("lnS")
    # plt.legend()
    # plt.show()

    if model.coef_ == 0:
        return 0
    if abs(model.coef_) < 0.01:  # 如果斜率接近于零，则T2值为零，否则出现很大值
        return 0
    else:
        T2 = -1/model.coef_  # 由斜率求出T2
        return round(float(T2), 3)


def get_T2_mapping():
    # 正常对照组
    # root_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    # 有病的组
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\split_data_ACLR_T2"
    dicom_folder_name = ["folder_" + str(i) for i in range(1, 6)]
    nii_folder_name = [i + "_nii" for i in dicom_folder_name]
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        dicom_folder_path = [os.path.join(case_path, i) for i in dicom_folder_name]
        nii_folder_path = [os.path.join(case_path, i) for i in nii_folder_name]

        # 从dicom图像中得到TE值
        one_case_TE = [get_one_folder_TE(i) for i in dicom_folder_path]
        TE_list = one_case_TE[1:]  # 去除掉第一个回波
        one_case_nii_array = [get_nii_array(i) for i in nii_folder_path]
        data_shape = get_data_shape(folder_path=nii_folder_path[0])
        T2_list = []
        for p_1, p_2, p_3, p_4, p_5 in zip(one_case_nii_array[0],
                                           one_case_nii_array[1],
                                           one_case_nii_array[2],
                                           one_case_nii_array[3],
                                           one_case_nii_array[4]):
            T2 = compute_t2_by_four_points(p_2, p_3, p_4, p_5, TE_list)
            T2_list.append(T2)
            print(T2)
        T2_array = np.asarray(T2_list)
        T2_array = np.reshape(T2_array, data_shape)
        np.save(os.path.join(case_path, "T2_mapping.npy"), T2_array)


def show_mapping():
    """显示T2_mapping的图像"""
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\normal_control"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        t2_mapping_path = os.path.join(case_path, "T2_mapping.npy")
        t2_array = np.load(t2_mapping_path)
        show_array(t2_array)


def check_TE():
    """检查有病的那一组的TE"""
    """检查后发现是正确的，是按照回波的顺序来存的"""
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\split_data_ACLR_T2"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        folder_name = ["folder_" + str(i) for i in range(1, 6)]
        for folder in folder_name:
            folder_path = os.path.join(case_path, folder)
            for file in os.listdir(folder_path):
                dicom_path = os.path.join(folder_path, file)
                _, TE, _ = get_TE(dicom_path)
                print(folder, TE)
        break


if __name__ == '__main__':
    # 通过五个回波的dicom图像得到nii图像
    # get_nii_by_echo()
    # 计算出T2mapping的图像
    get_T2_mapping()
    # show_mapping()
    # 检查异常组的TE
    # check_TE()
