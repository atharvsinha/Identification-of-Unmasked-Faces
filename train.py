from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

def Train():
    data = pickle.loads(open('embeddings/embeddings.pickle', "rb").read())
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open('embeddings/recognize.pickle', "wb+")
    pickle.dump(recognizer, f)
    f.close()
    # write the label encoder to disk
    f = open('embeddings/le.pickle', "wb+")
    pickle.dump(le, f)
    f.close()

def TrainCombo():
    data = pickle.loads(open('embeddings/embeddingscombo.pickle', "rb").read())
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open('embeddings/recognizecombo.pickle', "wb+")
    pickle.dump(recognizer, f)
    f.close()
    # write the label encoder to disk
    f = open('embeddings/lecombo.pickle', "wb+")
    pickle.dump(le, f)
    f.close()

def TrainMask():
    data = pickle.loads(open('embeddings/embeddingsmask.pickle', "rb").read())
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)  

    f = open('embeddings/recognizemask.pickle', "wb+")
    pickle.dump(recognizer, f)
    f.close()
    # write the label encoder to disk
    f = open('embeddings/lemask.pickle', "wb+")
    pickle.dump(le, f)
    f.close()
Train()
TrainCombo()
TrainMask()
