import os
import numpy as np
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import get_array_from_path, standard
import SimpleITK as sitk

def get_data_roi():
    root_path = r"Y:\DYB\PETCT_EGFR"
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            ct_path = os.path.join(case_path, "ct.nii")
            roi_path = os.path.join(case_path, "no_name.nii")
            data = sitk.ReadImage(ct_path)
            ct_array = get_array_from_path(ct_path)
            roi_array = get_array_from_path(roi_path)
            roi_array = np.flipud(roi_array)
            SaveArrayToNiiByRef(store_path=os.path.join(case_path, "ct_roi.nii"),
                                array=roi_array,
                                ref_image=data)
            print(case)


get_data_roi()