import numpy as np
import imutils
import pickle
import time
import cv2
import os

def Recog():
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models/res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models/deploy.prototxt.txt')

    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    detector = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

    recognizer = pickle.loads(open('embeddings/recognize.pickle', "rb").read())
    le = pickle.loads(open('embeddings/le.pickle', "rb").read())

    video = cv2.VideoCapture(0)
    while(True):
        _ , img = video.read(0)
        (h, w) = img.shape[:2]
        temp_img = cv2.resize(img, (w//2, h//2))
        blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.5,(300,300),(104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()
        k = cv2.waitKey(1)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >.5:
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                if 'unknown' in text:
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 1)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                else:
                    cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 255), 1)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
        cv2.imshow("Face Recognition", img)
        if k%256==27:
                break    
    video.release() 
    cv2.destroyAllWindows()

def RecogMask():
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models/res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models/deploy.prototxt.txt')

    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    detector = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

    recognizer = pickle.loads(open('embeddings/recognizemask.pickle', "rb").read())
    le = pickle.loads(open('embeddings/lemask.pickle', "rb").read())

    video = cv2.VideoCapture(0)
    while(True):
        _ , img = video.read(0)
        (h, w) = img.shape[:2]
        temp_img = cv2.resize(img, (w//2, h//2))
        blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.5,(300,300),(104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()
        k = cv2.waitKey(1)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >.5:
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                if 'no' in text:
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 1)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                else:
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 1)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("Face Recognition", img)
        if k%256==27:
                break    
    video.release() 
    cv2.destroyAllWindows()

def RecogCombo():
    caffeModel = os.path.dirname(os.path.realpath(__file__))
    caffeModel = os.path.join(caffeModel, 'models/res10_300x300_ssd_iter_140000.caffemodel')
    prototextPath = os.path.dirname(os.path.realpath(__file__))
    prototextPath = os.path.join(prototextPath, 'models/deploy.prototxt.txt')

    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    detector = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

    recognizer = pickle.loads(open('embeddings/recognizecombo.pickle', "rb").read())
    le = pickle.loads(open('embeddings/lecombo.pickle', "rb").read())

    video = cv2.VideoCapture(0)
    unknown_count=0
    while(True):
        _ , img = video.read(0)
        (h, w) = img.shape[:2]
        temp_img = cv2.resize(img, (w//2, h//2))
        blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.5,(300,300),(104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()
        k = cv2.waitKey(1)
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >.5:
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                
                if 'unknown' in text:
                    unknown_count+=1
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 1)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    if  unknown_count%10==0:
                        os.system(f'play -nq -t alsa synth {.1} sine {440}')

                elif 'mask' in text:
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 1)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 255), 1)
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                
        cv2.imshow("Face Recognition", img)
        if k%256==27:
                break    
    video.release() 
    cv2.destroyAllWindows()