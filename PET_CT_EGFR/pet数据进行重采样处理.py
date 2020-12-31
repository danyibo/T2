import os
import numpy as np
import SimpleITK as stik
import matplotlib.pyplot as plt
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
from Tool.CT_dcm_to_nii import dcm_to_nii
from ALL.MIP4AIM.MIP4AIM.Dicom2Nii.Dicom2Nii import ProcessSeries

class GetPetData:
    def __init__(self, case_path):
        self.case_path = case_path

    def remove_no_use_pet(self):
        """删除没用的pet.nii"""
        bad_file_path = os.path.join(self.case_path, "pet", "pet.nii")
        try:
            os.remove(bad_file_path)
        except:
            pass

    def reanme(self):
        """将ct，pet 的文件夹改名字"""
        pass

    def check_ct_pet_nii(self):
        case_file_list = os.listdir(self.case_path)
        if "pet.nii" and "ct.nii" not in case_file_list:
            print(case_path)

    def check_resample_result(self):
        """检查重采样的结果"""
        resample_result = os.path.join(self.case_path, "pet_Resize.nii")
        pet_array = get_array_from_path(resample_result)
        sum_array = np.sum(pet_array)
        pet_array = standard(pet_array)
        roi_path = os.path.join(self.case_path, "no_name.nii")
        roi_array = get_array_from_path(roi_path)
        pet_array = np.flipud(pet_array)
        Imshow3DArray(pet_array, roi_array)

        print(sum_array)


if __name__ == '__main__':
    root_path = r"Y:\DYB\PETCT_EGFR"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            # 实例化类对象
            get_pet_data = GetPetData(case_path=case_path)
            # (1): 删除没用的文件
            # get_pet_data.remove_no_use_pet()
            # (2): 进行转换
            # get_pet_data.reanme()
            # (3): 检查文件数量
            # get_pet_data.check_ct_pet_nii()
            # (4): 检查重采样之后的结果，看看有没有为零的情况
            # get_pet_data.check_resample_result()
            # (5): 检查crop后的结果

            ################################
            # 重新转了一次大小不匹配的图像CT #
            ################################

            # import SimpleITK as sitk
            # roi = sitk.ReadImage(os.path.join(case_path, "no_name.nii"))
            # for file in os.listdir(case_path):
            #     if file[0] == "C":
            #         ct_folder = os.path.join(case_path, file, "ct")
            #         from MeDIT.SaveAndLoad import SaveArrayToNiiByRef
            #         reader = sitk.ImageSeriesReader()
            #         dcm_names = reader.GetGDCMSeriesFileNames(ct_folder)
            #
            #         reader.SetFileNames(dcm_names)
            #         dcm_image_all = reader.Execute()
            #         dcm_image_array = sitk.GetArrayFromImage(dcm_image_all)
            #         dcm_image_array = np.transpose(dcm_image_array, (1, 2, 0))
            #         dcm_image_array = np.flipud(dcm_image_array)
            #         SaveArrayToNiiByRef(store_path=os.path.join(case_path, "ct.nii"),
            #                             array=dcm_image_array,
            #                             ref_image=roi)
            #         print(case)




