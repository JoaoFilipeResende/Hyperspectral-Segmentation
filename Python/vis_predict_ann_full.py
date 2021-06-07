from vis_generator_ann_full import VisGenerator
import os
import tensorflow as tf
import numpy as np
from hyperimage import Hyperimage

from sklearn.metrics import classification_report
from matplotlib import pyplot as plt, colors as pltcolors

MODEL_PATHS = ["vis_model_full_500_8f.h5", "vis_model_full_500_16f.h5"]

BATCH_SIZE = 1

TRAIN_PERCENTAGE = 0.7
VALIDATE_PERCENTAGE = 0.1
PREDICT_PERCENTAGE = 0.2


def process_prediction(prediction):
    output_img = np.empty([254, 510])
    for y in range(254):
        for x in range(510):
            output_img[y][x] = np.round(np.average(prediction[y][x]))
    return output_img


def get_true_values(file_list):
    hyper_images = []
    for idx in range(len(file_list)):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "vis", file_list[idx])
        hyper_images.append(Hyperimage(path))

    labels = []
    for img in hyper_images:
        labels.append(img.get_label().flatten())
    return np.asarray(labels)


if __name__ == '__main__':
    vis_gen_predict = VisGenerator(BATCH_SIZE, TRAIN_PERCENTAGE, VALIDATE_PERCENTAGE, PREDICT_PERCENTAGE, "predict")

    for model_name in MODEL_PATHS:
        model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name)
        model = tf.keras.models.load_model(model_name)

        predictions = model.predict_generator(generator=vis_gen_predict, verbose=1)

        file_list = vis_gen_predict.get_file_list()

        # Flatten predictions
        preds = list()
        for p in predictions:
            p = process_prediction(p)
            preds.append(p.flatten())
        y_predict = np.asarray(preds).flatten()

        y_real = get_true_values(file_list).flatten()

        label_names = ["Undefined", "Road", "Sidewalk", "Lane Markers", "Grass", "Vegetation",
                       "Panels/Signs/Traffic Lights", "Building/Wall/Others", "Car/Truck/Train/Bus...",
                       "Pedestrian/Cyclist/Motocyclist/Animal", "Sky"]

        # Print metrics
        output_file = open(str(model_name + "_metrics.txt"), 'w')
        report = classification_report(y_real, y_predict, labels=range(len(label_names)), target_names=label_names, zero_division=0)
        print(report)
        print(report, file=output_file)
        output_file.close()

        for idx in range(len(file_list)):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "vis", file_list[idx])
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
            axarr[1].imshow(process_prediction(predictions[idx]), cmap=pltcolors.ListedColormap(myColors), vmin=0,
                            vmax=10)
            plt.show(block=True)
