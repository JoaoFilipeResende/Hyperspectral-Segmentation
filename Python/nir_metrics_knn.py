import numpy as np
import os
from hyperimage import Hyperimage
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt, colors as pltcolors

TRAIN_PERCENTAGE = 0.8

RESULTS_FILE = "nir_knn_y_predict_3K.npy"
OUTPUT_METRICS_FILE = "nir_knn_3K_metrics.txt"


def process_prediction(y_predict, img_idx):
    output_img = np.empty([214, 407])
    for y in range(214):
        for x in range(407):
            img_offset = img_idx * 407 * 214
            px_idx = (y * 407 + x) + img_offset
            output_img[y][x] = y_predict[px_idx]
    return output_img


if __name__ == '__main__':

    file_list = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "nir"))
    file_list.sort()

    n_train_files = int(np.floor(TRAIN_PERCENTAGE * len(file_list)))
    train_files = file_list[:n_train_files]
    predict_files = file_list[n_train_files:]

    predict_img = []

    for filename in predict_files:
        predict_img.append(Hyperimage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2",
                                                   "nir", filename)))

    y_real = []
    for idx, img in enumerate(predict_img):
        for y in range(img.get_hypercube().shape[0]):
            for x in range(img.get_hypercube().shape[1]):
                y_real.append(img.get_label()[y, x])
    y_real = np.asarray(y_real)

    y_predict = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knn_results", RESULTS_FILE))

    label_names = ["Road", "Sidewalk", "Lane Markers", "Grass", "Vegetation",
                   "Panels/Signs/Traffic Lights", "Building/Wall/Others", "Car/Truck/Train/Bus...",
                   "Pedestrian/Cyclist/Motocyclist/Animal", "Sky"]

    output_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knn_results", OUTPUT_METRICS_FILE),
                       'w')
    undefined_px_idx = np.where(y_real == 0)
    filtered_y_real = np.delete(y_real, undefined_px_idx)
    filtered_y_predict = np.delete(y_predict, undefined_px_idx)
    report = classification_report(filtered_y_real, filtered_y_predict, labels=range(1, len(label_names)+1),
                                   target_names=label_names, zero_division=0)
    print(report)
    print(report, file=output_file)
    output_file.close()

    for idx in range(len(predict_files)):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "nir", predict_files[idx])
        img = Hyperimage(path)
        myColors = ["#ffffff",  # id: 0, name: undefined, (0,0,0)
                    "#ff0000",  # id: 1, name: Road (Ground - Drivable), (255,0,0)
                    "#ff8000",  # id: 2, name: Sidewalk (Ground - Drivable), (255,128,0)
                    "#ff00ff",  # id: 3, name: Lane Markers (Ground - Drivable), (255,0,255)
                    "#00ff00",  # id: 4, name: Grass (Ground - Drivable), (0,255,0)
                    "#00ff80",  # id: 5, name: Vegetation (Not Grass - Not Drivable), (0,255,128)
                    "#5500dc",  # id: 6, name: Panels/Signs/TraficLight (Static Obstacles), (85,0,220)
                    "#55009d",  # id: 7, name: Building/Wall/Others (Static Obstacles, (85,0,157)
                    "#ffff00",  # id: 8, name: Car/Truck/Train/Bus/Plane/Bicycle/Motocycle/etc., (255,255,0)
                    "#ffaac0",  # id: 9, name: Adult,Children,Cyclist,Motocyclist,Animal - (Person - Moveable Obstacle), (255,170,192)
                    "#0000ff"]  # id: 10, name: Sky, (0,0,255)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img.get_label(), cmap=pltcolors.ListedColormap(myColors), vmin=0, vmax=10)
        axarr[1].imshow(process_prediction(y_predict, idx), cmap=pltcolors.ListedColormap(myColors), vmin=0,
                        vmax=10)
        plt.show(block=True)
