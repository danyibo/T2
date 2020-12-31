import os
import numpy as np
import matplotlib.pyplot as plt
from ALL.MIP4AIM.MIP4AIM.Dicom2Nii.Dicom2Nii import ProcessSeries
import SimpleITK as sitk
import shutil
from Tool.FolderProcress import make_folder


def get_all_dicom():
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\split_data_ACLR_T2"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        store_all_folder = make_folder(os.path.join(case_path, "all_dicom"))
        for folder in os.listdir(case_path):
            folder_path = os.path.join(case_path, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                shutil.copy(file_path, store_all_folder)
        print(case)

def get_nii():
    root_path = r"Y:\DYB\2020832DATA\doctor_xie\split_data_ACLR_T2"
    for case in os.listdir(root_path):
        all_nii_path = make_folder(os.path.join(root_path, case, "nii_folder"))
        all_folder_path = os.path.join(root_path, case, "all_dicom")
        # ProcessSeries(all_folder_path, all_nii_path)
        num_of_nii = len(os.listdir(all_nii_path))
        if num_of_nii != 5:
            print(case)


# get_all_dicom()
get_nii()