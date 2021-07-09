import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle
import cv2
import numpy as np

DATA_FOLDER = '/Users/nhanphan/CoderSchool/opencv/facial_detection/model/data'

def train():
    # Load and preprocess images and labels
    p = Path(DATA_FOLDER)
    all_paths = list(p.glob('**/*.jpg'))
    all_labels = list(map(lambda x: str(x).split('/')[-2], all_paths))

    label_dict = {}
    for i, name in enumerate(set(all_labels)):
        label_dict[i] = name

    labels = label_binarize(all_labels, classes=list(set(all_labels)))

    def load_and_preprocess(path):
        path = str(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
        img = img / 255.
        img = img.flatten()

        return img

    images = np.array(list(map(load_and_preprocess, all_paths)))

    # Training 
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(images, labels)
    y_pred = knn.predict(images)
    print('ACCURACY: ', accuracy_score(labels, y_pred))

    # Save model and label set
    _file = open("knn.pkl", "wb")
    pickle.dump(knn, _file)
    _file.close()

    a_file = open("labels.pkl", "wb")
    pickle.dump(label_dict, a_file)
    a_file.close()

    print('FINISHED TRAINING')

if __name__ == '__main__':
    train()
