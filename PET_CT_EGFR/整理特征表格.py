import os
import pandas as pd



class GetCsv:
    def __init__(self):
        self.root_path = r"Y:\DYB\2020832DATA\pet_ct"
        self.ct_yin = os.path.join(self.root_path, "ct_yin.csv")
        self.ct_yang = os.path.join(self.root_path, "ct_yang.csv")
        self.pet_yin = os.path.join(self.root_path, "pet_yin.csv")
        self.pet_yang = os.path.join(self.root_path, "pet_yang.csv")

    def get_train_test(self):
         ct_yin = pd.read_csv(self.ct_yin)
         ct_yang = pd.read_csv(self.ct_yang)

get_csv = GetCsv()
get_csv.get_train_test()
