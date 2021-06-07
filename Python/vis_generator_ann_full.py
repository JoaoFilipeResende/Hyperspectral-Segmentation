import tensorflow as tf
import os
import numpy as np
from hyperimage import Hyperimage


class VisGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size, train_percent, validate_percent, predict_percent, gen_type):
        self.batch_size = batch_size
        self.train_percent, self.validate_percent, self.predict_percent = train_percent, validate_percent, predict_percent
        if (gen_type != "train") and (gen_type != "validate") and (gen_type != "predict"):
            raise Exception('gen_type must be "train", "validate" or "predict"')
        self.gen_type = gen_type
        self.file_list = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "vis"))
        self.file_list.sort()

        train_size = int(np.ceil(train_percent * len(self.file_list)))
        validate_size = int(np.floor(validate_percent * len(self.file_list)))

        if self.gen_type == "train":
            self.file_list = self.file_list[:train_size]
        elif self.gen_type == "validate":
            self.file_list = self.file_list[train_size:train_size + validate_size]
        else:
            self.file_list = self.file_list[train_size + validate_size:]

    def __len__(self):
        return np.ceil(len(self.file_list) / float(self.batch_size)).astype(np.int)

    def __getitem__(self, batch_idx):
        hyper_images = []
        batch_file_list = self.file_list[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        for filename in batch_file_list:
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "vis", filename)
            hyper_images.append(Hyperimage(img_path))

        x_train = np.asarray([img.get_hypercube() for img in hyper_images])
        x_train = x_train[..., np.newaxis]

        y_train = []
        for img in hyper_images:
            labels = np.repeat(img.get_label()[:, :, np.newaxis], 15, axis=2)
            y_train.append(labels)

        y_train = np.asarray(y_train)

        return x_train, y_train

    def get_file_list(self):
        return self.file_list
