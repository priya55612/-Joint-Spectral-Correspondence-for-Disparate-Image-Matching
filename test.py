import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def loadSymbench(path):
    folders = os.walk(path)
    for folder in folders:
        print(folder.name)
    pass

if __name__ == "__main__":
    loadSymbench('dataset/symbench_v1/')
    pass