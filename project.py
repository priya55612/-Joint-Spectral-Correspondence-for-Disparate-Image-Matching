import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadSymbench():
    data_root = 'dataset\symbench_v1'
    i1 = '01.png'
    i2 = '02.png'
    h = 'H1to2'
    symbench = {}
    for dirname,folder,files in os.walk(data_root):
        if(dirname==data_root):
            continue
        record = {}
        folder_name = dirname.split('\\')[-1]
        h_path = str(dirname)+'\\'+h
        i1_path = str(dirname)+'\\'+i1
        i2_path = str(dirname)+'\\'+i2
        with open(h_path,'r') as f:
            record['h'] = np.array(f.read())
        record['i1'] = cv2.imread(i1_path)
        record['i2'] = cv2.imread(i2_path)
        symbench[folder_name] = record
    return symbench
    pass

if __name__ == "__main__":
    symbench = loadSymbench()
    for key in symbench:
        print(symbench[key]['i1'])
    pass