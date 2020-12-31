"""
 2020-11-6
"""
import os
import shutil
import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd

from Tool.DataProcress import get_array_from_path, standard
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef
from Tool.FolderProcress import make_folder
from MeDIT.Visualization import Imshow3DArray
from MIP4AIM.NiiProcess.Registrator import Registrator


####################################################################
#   整理数据
#  1. 整理 data roi
#  2. 整理 pet 数据 需要放缩一下
#  3. 整理临床数据
#  4. 提取特征后进行匹配
#  5. 目前的处理方法是将pet数据进行放大处理了，不知道这种方法是否可以
####################################################################


###################
# 先查看CT数据
###################

class CheckCTData:
    def __init__(self):
        self.egfr_yang_path = r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\EGFR+new_C_CT"
        self.egfr_yin_path = r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\EGFR-new_C_CT"

    def check_data_roi(self):
        """检查"""
        pass


#################
#  处理临床特征
#################


def rename_clinical_case(pd_csv, label):
    """label 是阳性和阴性的类别"""
    case_name = pd_csv["CaseName"]
    case_name = [str(i) for i in case_name]
    new_case_name = []
    for i in case_name:
        if len(i) == 1:
            new_case_name.append("00"+i)
        elif len(i) == 2:
            new_case_name.append("0"+i)
        elif len(i) == 3:
            new_case_name.append(i)
        else:
            new_case_name.append(i)

    if label == 0:
        new_case_name = [i+"-" for i in new_case_name]
        pd_csv["CaseName"] = new_case_name
        return pd_csv
    else:
        new_case_name = [i+"+" for i in new_case_name]
        pd_csv["CaseName"] = new_case_name
        return pd_csv


def get_clinical_feature():
    clinical_yin = r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\clinical_yin.csv"
    clinical_yang = r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\clinical_yang.csv"
    pd_yin = pd.read_csv(clinical_yin, encoding='gb18030')
    pd_yang = pd.read_csv(clinical_yang, encoding='gb18030')
    # 进行修改名字
    new_pd_yin = rename_clinical_case(pd_csv=pd_yin, label=0)
    new_pd_yang = rename_clinical_case(pd_csv=pd_yang, label=1)
    # 得到修改名字后的临床特征表格
    clinical_feature = new_pd_yin.append(new_pd_yang)

    clinical_feature.to_csv(os.path.join(
        r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\Clinical", "clinical_feature.csv"), index=None)
# get_clinical_feature()

##################################
#   将临床特征和组学特征进行融合
##################################


def get_combine_feature():
    clinical_path = r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\Clinical\clinical_feature.csv"
    feature_path = r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\feature\gbdt_train_test.csv"
    pd_clinical = pd.read_csv(clinical_path)
    pd_feature = pd.read_csv(feature_path)
    pd_result = pd.merge(pd_clinical, pd_feature, how="left", on="CaseName")
    new_result = pd_result.dropna(how="any", axis=0)
    new_result.to_csv(os.path.join(r"Y:\DYB\2020832DATA\doctor_gao\PECTCT_EGFR\feature", "clinical_and_feature.csv"), index=None)

# get_combine_feature()


###################
# PET数据合成nii
###################


class PetToNii:
    def __init__(self, dicom_path, roi_path, store_path):
        self.dicom_path = dicom_path  # 一个case, 存放dicom文件的路径
        self.roi_path = roi_path
        self.roi_array = get_array_from_path(self.roi_path)
        self.roi = sitk.ReadImage(self.roi_path)
        self.store_path = store_path
        self.oirgin_roi_array = sitk.GetArrayFromImage(sitk.ReadImage(self.roi_path))
        print(self.oirgin_roi_array.shape)

    def get_nii(self):
        reader = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(self.dicom_path)

        reader.SetFileNames(dcm_names)
        dcm_image_all = reader.Execute()
        dcm_image_array = sitk.GetArrayFromImage(dcm_image_all)
        dcm_result = sitk.GetImageFromArray(dcm_image_array)
        sitk.WriteImage(dcm_result, os.path.join(self.store_path, "origin_pet.nii"))


def run_pet_to_nii():
    root_path = r"Y:\DYB\PETCT_EGFR\EGFR-"
    for case in os.listdir(root_path):
        roi_path = os.path.join(root_path, case, "no_name.nii")
        pet_path = os.path.join(root_path, case, "P", "images")
        store_path = make_folder(os.path.join(root_path, case, "pet"))
        pet_to_nii = PetToNii(dicom_path=pet_path,
                              roi_path=roi_path,
                              store_path=store_path)
        pet_to_nii.get_nii()
        print("case {} is finished!".format(case))
        break

# run_pet_to_nii()


#################################
# CT数据合成NII 也存在pet文件夹中
#################################
class CTToNii:
    def __init__(self, dicom_path, roi_path, store_path):
        self.dicom_path = dicom_path  # 一个case, 存放dicom文件的路径
        self.roi_path = roi_path
        self.roi_array = get_array_from_path(self.roi_path)
        self.roi = sitk.ReadImage(self.roi_path)
        self.store_path = store_path

    def get_nii(self):
        reader = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(self.dicom_path)
        reader.SetFileNames(dcm_names)
        dcm_image_all = reader.Execute()
        dcm_image_array = sitk.GetArrayFromImage(dcm_image_all)
        dcm_image_array = np.transpose(dcm_image_array, (1, 2, 0))
        try:
            SaveArrayToNiiByRef(store_path=os.path.join(self.store_path, "ct.nii"),
                                array=dcm_image_array,
                                ref_image=self.roi)
        except:
            pass


def run_CT_to_nii(root_path):
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        for folder in os.listdir(case_path):
            if folder not in ["P", "pet", "no_name.nii"]:
                ct_folder = os.path.join(case_path, folder, "images")
                roi_path = os.path.join(root_path, case, "no_name.nii")
                store_path = make_folder(os.path.join(root_path, case, "pet"))
                ct_to_nii = CTToNii(dicom_path=ct_folder, roi_path=roi_path, store_path=store_path)
                ct_to_nii.get_nii()
                print("case {} is finished!".format(case))


# run_CT_to_nii(root_path=r"Y:\DYB\PETCT_EGFR\EGFR-")
# run_CT_to_nii(root_path=r"Y:\DYB\PETCT_EGFR\EGFR+")


#####################################
# 将PET CT roi整理到一个标准的文件夹下
#####################################

class GetCtPetRoi:
    def __init__(self, root_path, save_folder_name):
        self.root_path = root_path
        self.store_path = r"Y:\DYB\2020832DATA\doctor_gao\pet_ct_egfr"
        self.save_folder_path = make_folder(os.path.join(self.store_path, save_folder_name))

    def get_ct_pet_roi(self):
        problem_case = []
        for case in os.listdir(self.root_path):
            ct_path = os.path.join(self.root_path, case, "pet", "ct.nii")
            pet_path = os.path.join(self.root_path, case, "pet", "pet.nii")
            roi_path = os.path.join(self.root_path, case, "no_name.nii")
            if len(os.listdir(os.path.join(self.root_path, case, "pet"))) == 2:
                save_case_path = make_folder(os.path.join(self.save_folder_path, case))
                try:
                    shutil.copy(ct_path, save_case_path)
                    shutil.copy(pet_path, save_case_path)
                    shutil.copy(roi_path, save_case_path)
                except:
                    problem_case.append(case)
            print("case {} is finished!".format(case))
        print("###########################")
        print(problem_case)
        print("###########################")


def run_get_ct_pet_roi(root_path, save_folder_name):
    get_ct_pet_roi = GetCtPetRoi(root_path=root_path, save_folder_name=save_folder_name)
    get_ct_pet_roi.get_ct_pet_roi()


# run_get_ct_pet_roi(root_path=r"Y:\DYB\PETCT_EGFR\EGFR+", save_folder_name="EGFR+")
# run_get_ct_pet_roi(root_path=r"Y:\DYB\PETCT_EGFR\EGFR-", save_folder_name="EGFR-")

######################
#  显示提取特征
######################


def show():
    """经过显示发现图像并不是很好，可能需要配准一下"""
    root_path = r"Y:\DYB\2020832DATA\doctor_gao\pet_ct_egfr\EGFR-"
    for case in os.listdir(root_path):
        ct_path = os.path.join(root_path, case, "ct.nii")
        pet_path = os.path.join(root_path, case, "pet.nii")
        roi_path = os.path.join(root_path, case, "no_name.nii")
        ct_array = get_array_from_path(ct_path)
        pet_array = get_array_from_path(pet_path)
        roi_array = get_array_from_path(roi_path)
        Imshow3DArray(standard(ct_array), roi_array)
        Imshow3DArray(standard(pet_array), roi_array)

# show()

#####################
#   进行配准
#####################


def Registrator_image(fix_path, moving_path):
    """
    得到配准后的结果，reg 是矩阵，result 是 配准后的图像nii
    :param fix_path:
    :param moving_path:
    :return:
    """
    registrator = Registrator()
    registrator.fixed_image = fix_path
    registrator.moving_image = moving_path

    result = registrator.RegistrateBySpacing()
    reg = np.asarray(sitk.GetArrayFromImage(result), dtype=np.float32)
    reg = np.transpose(reg, (1, 2, 0))
    return reg, result


def get_new_pet(root_path):
    for case in os.listdir(root_path):
        pet_path = os.path.join(root_path, case, "pet.nii")
        ct_path = os.path.join(root_path, case, "ct.nii")
        roi_path = os.path.join(root_path, case, "no_name.nii")
        roi_array = get_array_from_path(roi_path)

        reg, result = Registrator_image(fix_path=ct_path, moving_path=pet_path)
        Imshow3DArray(standard(reg), roi_array)


# get_new_pet(root_path=r"Y:\DYB\2020832DATA\doctor_gao\pet_ct_egfr\EGFR+")


#######################
#   查看重采样后的图像
#######################
import matplotlib.pyplot as plt



def show_resample_image():
    data_path = r"Y:\DYB\PETCT_EGFR\EGFR-\001-\pet_Resize.nii"
    roi_path = r"Y:\DYB\PETCT_EGFR\EGFR-\001-\no_name.nii"
    roi_array = get_array_from_path(roi_path)

    data_array = get_array_from_path(data_path)
    data_array = np.flipud(data_array)
    ct_path = r"Y:\DYB\PETCT_EGFR\EGFR-\001-\ct.nii"
    ct_array = get_array_from_path(ct_path)
    ct_pet_array = ct_array + data_array
    ct_pet_array = standard(ct_pet_array)
    ct_pet_array = np.flipud(ct_pet_array)
    ct_array = np.flipud(ct_array)
    # Imshow3DArray(standard(ct_array), roi_array)
    Imshow3DArray(ct_pet_array, roi_array)
    # print(np.sum(data_array))
    # data_array = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
    # print(data_array.shape)
    # print(np.sum(data_array))
    # for i in range(data_array.shape[-1]):
    #     if np.sum(roi_array[..., i]) != 0:
    #         plt.imshow(data_array[..., i], cmap="gray")
    #         plt.contour(roi_array[..., i], colors="red")
    #         plt.show()

    # print(data_array.shape, roi_array.shape)
    # # # data_array = np.transpose(data_array, (1, 2, 0))
    # # data_array = standard(data_array)
    # # Imshow3DArray(data_array, roi_array)

show_resample_image()