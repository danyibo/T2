import os
import numpy as np
from Tool.DataProcress import get_array_from_path

case_path = r"Y:\DYB\data_and_result\doctor tao\data\all_data\Ankle instability\DICOMDIS"
roi_name_list = ["roi_" + str(i) for i in range(1, 9)]
for roi in roi_name_list:
    roi_path = os.path.join(case_path, roi)
    roi_array = get_array_from_path(roi_path)
    print(np.unique(roi_array))

