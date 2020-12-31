import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import standard
from MeDIT.ArrayProcess import SmoothRoi, RemoveSmallRegion, KeepLargestRoi
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef


class SaveNumpyToNii:
    """将数据存为nii"""
    def __init__(self, case_path):
        self.case_path = case_path
        self.ct_path = os.path.join(self.case_path, "croped_ct.npy")
        self.pet_path = os.path.join(self.case_path, "croped_pet.npy")
        self.roi_path = os.path.join(self.case_path, "result_roi.npy")
        self.ct_array = np.load(self.ct_path)
        self.pet_array = np.load(self.pet_path)
        self.roi_array = np.load(self.roi_path)
        self.roi_array = KeepLargestRoi(RemoveSmallRegion(SmoothRoi(self.roi_array)))
        self.cp_array = self.ct_array + self.pet_array

    def show_roi(self):
        """检查数据展示几个ROI"""
        ct_array = standard(self.pet_array)
        Imshow3DArray(ct_array, self.roi_array)

    def save(self):
        ref = sitk.GetImageFromArray(self.ct_array)
        SaveArrayToNiiByRef(os.path.join(self.case_path, "result_ct.nii.gz"),
                            np.transpose(self.ct_array, (1, 2, 0)),
                            ref)
        SaveArrayToNiiByRef(os.path.join(self.case_path, "result_pet.nii.gz"),
                            np.transpose(self.pet_array, (1, 2, 0)),
                            ref)
        SaveArrayToNiiByRef(os.path.join(self.case_path, "result_cp.nii.gz"),
                            np.transpose(self.cp_array, (1, 2, 0)),
                            ref)
        SaveArrayToNiiByRef(os.path.join(self.case_path, "result_roi.nii.gz"),
                            np.transpose(self.roi_array, (1, 2, 0)),
                            ref)


if __name__ == '__main__':
    root_path = r"Y:\DYB\2020832DATA\doctor_gao\PET_EGFR\EGFR_PET_CT_DATA"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            try:
                save = SaveNumpyToNii(case_path)
                save.save()
                print(case)
            except: pass




