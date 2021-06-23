import os
import cv2
import pickle
import numpy as np
from imutils import paths
def RecogFromMask(img):
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models/res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models/deploy.prototxt.txt')

    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    detector = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

    recognizer = pickle.loads(open('embeddings/recognizecombo.pickle', "rb").read())
    le = pickle.loads(open('embeddings/lecombo.pickle', "rb").read())
    
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1,(300,300),(104.0, 177.0, 123.0, ))
    detector.setInput(blob)
    detections = detector.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = img[startY:endY, startX:endX]
            try:
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            except:
                continue
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{}: {:.2f}%".format(name, proba * 100)
            return text
                        


l1, l2, l3, l4, l5, l6, l7, l8, n1, n2, n3, n4, n5, n6, n7, n8 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

validate = os.path.join(os.path.dirname(__file__))
validate = os.path.join(validate, 'validate\\unknown\\')
imgs = list(paths.list_images(validate))
for ii, xx in enumerate(imgs):
    img = cv2.imread(xx)
    text = RecogFromMask(img)
    print(text)
    try:
        x, y  = text.split()
    except:
        continue
    if 'atharv' in x:
        n1+=1
        l1 += float(y[:-1])
    elif 'bill'in x:
        n2+=1
        l2 += float(y[:-1])
    elif 'trump'in x:
        n3+=1
        l3 += float(y[:-1])
    elif 'modi'in x:
        n4+=1
        l4 += float(y[:-1])
    elif 'ma'in x:
        n5+=1
        l5 += float(y[:-1])
    elif 'mask'in x:
        n6+=1
        l6 += float(y[:-1])
    elif 'unknown'in x:
        n7+=1
        l7 += float(y[:-1])
    elif 'elon'in x:
        n8+=1
        l8 += float(y[:-1])
    print(xx)
total = sum([l1, l2, l3, l4, l5, l6, l7, l8])
print(total)
print(l1/total, 'n1')
print(l2/total, 'n2')
print(l3/total, 'n3')
print(l4/total, 'n4')
print(l5/total, 'n5')
print(l6/total, 'n6')
print(l7/total, 'n7')
print(l8/total, 'n8')



