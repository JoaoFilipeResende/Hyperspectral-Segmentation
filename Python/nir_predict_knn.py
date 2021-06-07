from sklearn.neighbors import KNeighborsClassifier
import os
from hyperimage import Hyperimage
import numpy as np
import time

TRAIN_PERCENTAGE = 0.8
NEIGHBORS = 5

if __name__ == '__main__':

    knn = KNeighborsClassifier(n_neighbors=NEIGHBORS, n_jobs=None)

    file_list = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "nir"))
    file_list.sort()

    n_train_files = int(np.floor(TRAIN_PERCENTAGE * len(file_list)))
    train_files = file_list[:n_train_files]
    predict_files = file_list[n_train_files:]

    train_img = []
    predict_img = []

    for filename in train_files:
        train_img.append(Hyperimage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2",
                                                                                               "nir", filename)))

    for filename in predict_files:
        predict_img.append(Hyperimage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2",
                                                                                               "nir", filename)))

    # Setup training data
    x_train = []
    y_train = []
    for img in train_img:
        for y in range(img.get_hypercube().shape[0]):
            for x in range(img.get_hypercube().shape[1]):
                if img.get_label()[y, x] != 0: # Remove unlabeled pixels from training data
                    x_train.append(img.get_hypercube()[y, x])
                    y_train.append(img.get_label()[y, x])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    knn.fit(x_train, y_train)

    y_real = []
    x_predict = []
    y_predict = []

    for idx, img in enumerate(predict_img):
        for y in range(img.get_hypercube().shape[0]):
            for x in range(img.get_hypercube().shape[1]):
                x_predict.append(img.get_hypercube()[y, x])
                y_real.append(img.get_label()[y, x])
    x_predict = np.asarray(x_predict)
    y_real = np.asarray(y_real)

    start_time = time.time()
    y_predict = knn.predict(x_predict)
    print("Time taken:", time.time() - start_time)

    np.save(str("nir_knn_y_predict_" + str(NEIGHBORS) + "K.npy"), y_predict)
