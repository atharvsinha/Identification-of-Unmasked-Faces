from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
def Extraction():
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models\\res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models\\deploy.prototxt.txt')
    detect = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)
    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    dataset = os.path.dirname(os.path.realpath(__file__))
    dataset = os.path.join(dataset, 'dataset\\employees')
    img_path = list(paths.list_images(dataset))
    imageEmbeddings=[]
    names=[]
    total = 0
    for (i, path) in enumerate(img_path):
        name = (path.split('\\')[-1]).split('_')[0]
        image = cv2.imread(path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        Blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        detect.setInput(Blob)
        detections = detect.forward()
        if len(detections)==1:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > .9:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]   
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                names.append(name)
                imageEmbeddings.append(vec.flatten())
                total += 1
    data = {"embeddings": imageEmbeddings, "names": names}
    f = open("embeddings\\embeddings.pickle", "wb+")
    pickle.dump(data, f)
    f.close()
    

def ExtractionCombo():
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models\\res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models\\deploy.prototxt.txt')
    detect = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)
    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    dataset = os.path.dirname(os.path.realpath(__file__))
    dataset = os.path.join(dataset, 'dataset\\combo')
    img_path = list(paths.list_images(dataset))
    imageEmbeddings=[]
    names=[]
    total = 0
    for (i, path) in enumerate(img_path):
        name = (path.split('\\')[-1]).split('_')[0]
        image = cv2.imread(path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        Blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        detect.setInput(Blob)
        detections = detect.forward()
        if len(detections)==1:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > .5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]   
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                names.append(name)
                imageEmbeddings.append(vec.flatten())
                total += 1
    data = {"embeddings": imageEmbeddings, "names": names}
    f = open("embeddings\\embeddingscombo.pickle", "wb+")
    pickle.dump(data, f)
    f.close()
    
def ExtractionMask():
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models\\res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models\\deploy.prototxt.txt')
    detect = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)
    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    dataset = os.path.dirname(os.path.realpath(__file__))
    dataset = os.path.join(dataset, 'dataset\\maskmodule')
    img_path = list(paths.list_images(dataset))
    imageEmbeddings=[]
    names=[]
    total = 0
    for (i, path) in enumerate(img_path):
        name = (path.split('\\')[-1]).split('_')[0]
        image = cv2.imread(path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        Blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        detect.setInput(Blob)
        detections = detect.forward()
        if len(detections)==1:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > .5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]   
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                names.append(name)
                imageEmbeddings.append(vec.flatten())
                total += 1
    data = {"embeddings": imageEmbeddings, "names": names}
    
    f = open("embeddings\\embeddingsmask.pickle", "wb+")
    pickle.dump(data, f)
    f.close()

Extraction()
ExtractionMask()
ExtractionCombo()
