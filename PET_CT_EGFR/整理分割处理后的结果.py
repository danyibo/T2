import os
import numpy as np
import matplotlib.pyplot as plt
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import standard, get_array_from_path
import SimpleITK as sitk



class GetFinalResult:
    def __init__(self):
        self.root_path = r"Y:\DYB\PETCT_EGFR"

    def show_array(self, data_array, roi_array):
        for i in range(5, data_array.shape[-1]):
            if np.sum(roi_array[..., i]) != 0:
                plt.imshow(data_array[..., i], cmap="gray")
                plt.contour(roi_array[..., i], colors="red")
                plt.show()
                break


    def get_result(self):
        for folder in os.listdir(self.root_path):
            folder_path = os.path.join(self.root_path, folder)
            for case in os.listdir(folder_path):
                case_path = os.path.join(folder_path, case)
                ct_path = os.path.join(case_path, "croped_ct.npy")
                pet_path = os.path.join(case_path, "croped_pet.npy")
                roi_path = os.path.join(case_path, "pred_roi.npy")
                ct_array = np.load(ct_path)
                pet_array = np.load(pet_path)
                roi_array = np.load(roi_path)
                pet_ct_array = pet_array + ct_array
                #############
                #  显示结果
                #############
                roi = sitk.GetImageFromArray(roi_array)
                pet_array = np.transpose(pet_array, (1, 2, 0))
                ct_array = np.transpose(ct_array, (1, 2, 0))
                pet_ct_array = np.transpose(pet_ct_array, (1, 2, 0))
                roi_array = np.transpose(roi_array, (1, 2, 0))
                SaveArrayToNiiByRef(store_path=os.path.join(case_path, "result_ct.nii")
                                    , array=ct_array
                                    , ref_image=roi)
                SaveArrayToNiiByRef(store_path=os.path.join(case_path, "result_pet.nii")
                                    , array=pet_array
                                    , ref_image=roi)
                SaveArrayToNiiByRef(store_path=os.path.join(case_path, "result_ct_pet.nii")
                                    , array=pet_ct_array
                                    , ref_image=roi)
                SaveArrayToNiiByRef(store_path=os.path.join(case_path, "result_roi.nii")
                                    , array=roi_array
                                    , ref_image=roi)
                print("Case {} is finished!".format(case))






get_result = GetFinalResult()
get_result.get_result()