import tensorflow as tf
import os
import numpy as np
from hyperimage import Hyperimage
from keras.utils import to_categorical
import copy


class NirGenerator(tf.keras.utils.Sequence):

    def validate_batch(self, batch_idx):
        start_px = (batch_idx * self.batch_size)

        last_img_idx = int(start_px / (214 * 407))
        hyper_img = Hyperimage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "nir",
                                            self.file_list[last_img_idx]))

        valid_labels = []
        labels = hyper_img.get_label()

        n_px_in_batch = self.batch_size
        if batch_idx == len(self) - 1:
            n_px_in_batch = len(self.file_list) * 214 * 407 - start_px

        for i in range(n_px_in_batch):
            img_idx = int((start_px + i) / (214 * 407))
            px_idx = start_px % (214 * 407)

            if img_idx != last_img_idx:
                labels = hyper_img.get_label()
                last_img_idx = copy.copy(img_idx)

            y = int(px_idx / 407)
            x = int(px_idx % 407)

            if (labels[y, x] != 0):
                valid_labels.append(labels[y, x])

        valid_labels = np.asarray(valid_labels)
        if (valid_labels.shape == (0,)):
            self.invalid_batches.append(batch_idx)
        else:
            self.valid_batches.append(batch_idx)

    def __init__(self, batch_size, train_percent, validate_percent, predict_percent, gen_type, patch_size):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.train_percent, self.validate_percent, self.predict_percent = train_percent, validate_percent, predict_percent

        if (gen_type != "train") and (gen_type != "validate") and (gen_type != "predict"):
            raise Exception('gen_type must be "train", "validate" or "predict"')
        self.gen_type = gen_type

        self.file_list = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "nir"))
        self.file_list.sort()

        train_size = int(np.ceil(train_percent * len(self.file_list)))
        validate_size = int(np.floor(validate_percent * len(self.file_list)))

        if self.gen_type == "train":
            self.file_list = self.file_list[:train_size]
        elif self.gen_type == "validate":
            self.file_list = self.file_list[train_size:train_size + validate_size]
        else:
            self.file_list = self.file_list[train_size + validate_size:]

        self.valid_batches = []
        self.invalid_batches = []
        self.length = np.ceil((len(self.file_list) * 214 * 407 / float(self.batch_size))).astype(np.int)

        # Get valid batch indices
        if (self.gen_type != "predict"):
            for batch_idx in range(len(self)):
                self.validate_batch(batch_idx)

        # Update number of batches
        self.length = self.length - len(self.invalid_batches)

    def __len__(self):
        if (self.gen_type == "predict"):
            return np.ceil((len(self.file_list) * 214 * 407 / float(self.batch_size))).astype(np.int)
        else:
            return self.length

    def __getitem__(self, batch_idx):

        # Replace batch index when it's an invalid batch by using one of the leftover valid batches
        if (batch_idx in self.invalid_batches):
            position = self.invalid_batches.index(batch_idx)
            batch_idx = self.valid_batches[len(self.valid_batches) - position - 1]

        start_px = (batch_idx * self.batch_size)
        padding = int((self.patch_size - 1) / 2)

        # Code smell
        last_img_idx = int(start_px / (214 * 407))
        hyper_img = Hyperimage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2", "nir",
                                            self.file_list[last_img_idx]))
        cube = hyper_img.get_hypercube()
        padded_cube = np.pad(cube, [(padding, padding), (padding, padding), (0, 0)], mode='constant', constant_values=0)
        labels = hyper_img.get_label()

        x_train = []
        y_train = []

        n_px_in_batch = self.batch_size
        if (self.gen_type == "predict"):
            if batch_idx == len(self) - 1:
                n_px_in_batch = len(self.file_list) * 214 * 407 - start_px
        elif batch_idx == max(self.valid_batches):
            n_px_in_batch = len(self.file_list) * 214 * 407 - start_px

        for i in range(n_px_in_batch):
            img_idx = int((start_px + i) / (214 * 407))
            px_idx = start_px % (214 * 407)

            if img_idx != last_img_idx:
                hyper_img = Hyperimage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "HyKo2",
                                                    "nir", self.file_list[img_idx]))
                cube = hyper_img.get_hypercube()
                padded_cube = np.pad(cube, [(padding, padding), (padding, padding), (0, 0)], mode='constant',
                                     constant_values=0)
                labels = hyper_img.get_label()
                last_img_idx = copy.copy(img_idx)

            y = int(px_idx / 407)
            x = int(px_idx % 407)

            if self.gen_type == "predict" or (labels[y, x] != 0):  # Remove unlabeled data from training and validation
                pad_x = x + padding
                pad_y = y + padding
                x_train.append(padded_cube[pad_y - padding:pad_y + padding + 1, pad_x - padding:pad_x + padding + 1, :])
                y_train.append(labels[y, x])

        x_train = np.asarray(x_train)
        x_train = x_train[..., np.newaxis]
        y_train = to_categorical(np.asarray(y_train), num_classes=11)

        return x_train, y_train

    def get_file_list(self):
        return self.file_list