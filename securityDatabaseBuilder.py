from dataDnn import DetectorDnn
from pathlib import Path
from imutils import paths
import os
import cv2
def newFace():
    print("What is the name of the person to be added?")
    name = input()
    if ' ' in name:
        ind = name.index(" ")
        name = name[:ind]+'-'+name[ind+1:]
    name = name.lower()
    dataset = os.path.dirname(os.path.realpath(__file__))
    dataset1 = os.path.join(dataset, 'dataset/employees')
    dataset2 = os.path.join(dataset, 'dataset/combo')
    ndir1 = os.path.join(dataset1, name)
    ndir2 = os.path.join(dataset2, name)
    try:
        os.mkdir(ndir1)
        os.mkdir(ndir2)
    except:
        pass 
    DetectorDnn(ndir1, ndir2, name)
    return 
    
def Security_System_Builder():
    print("Security System Database Builder")
    print("Masked and No Masked Face Database Created")
    print("Hey, user!\nDo you want to add a new face to your 'Employee' database? [y/n]")
    k = input()
    if(k=='y' or k=='Y'):
        newFace()
dataset = os.path.dirname(os.path.realpath(__file__))
dataset = os.path.join(dataset, 'validate\\')
mods = os.listdir(dataset)

for i in os.listdir(dataset):
    print(len(list(paths.list_images(os.path.join(dataset, i)))))