from nir_generator_ann_patch import NirGenerator
import tensorflow as tf
from keras.layers import *
from keras.layers.normalization import BatchNormalization

BATCH_SIZE = 2 ** 15
EPOCHS = 500
PATIENCE = 1000
PATCH_SIZE = 3

TRAIN_PERCENTAGE = 0.7
VALIDATE_PERCENTAGE = 0.1
PREDICT_PERCENTAGE = 0.2


def create_model(input_shape):
    model = tf.keras.models.Sequential()

    model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding="same",
                     input_shape=(input_shape[0], input_shape[1], input_shape[2], 1)))
    model.add(BatchNormalization(axis=4))
    model.add(Dropout(0.2))

    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization(axis=4))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(11, activation="softmax"))

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == '__main__':
    nir_gen_train = NirGenerator(BATCH_SIZE, TRAIN_PERCENTAGE, VALIDATE_PERCENTAGE, PREDICT_PERCENTAGE, "train",
                                 PATCH_SIZE)
    nir_gen_validate = NirGenerator(BATCH_SIZE, TRAIN_PERCENTAGE, VALIDATE_PERCENTAGE, PREDICT_PERCENTAGE, "validate",
                                    PATCH_SIZE)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Early stopping callback to stop when validation accuracy stops improving
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.01, verbose=1, baseline=0.85,
                                          patience=PATIENCE)
    mcp_save = tf.keras.callbacks.ModelCheckpoint('nir_model_patch_checkpoint.h5', save_best_only=True,
                                                  monitor='val_accuracy',
                                                  mode='max')
    cb_list = [mcp_save]

    model = create_model((PATCH_SIZE, PATCH_SIZE, 25))

    model.fit(
        x=nir_gen_train,
        epochs=EPOCHS,
        verbose=1,
        callbacks=cb_list,
        validation_data=nir_gen_validate,
        steps_per_epoch=nir_gen_train.__len__(),
        validation_steps=nir_gen_validate.__len__()
    )

    model.save("nir_model_patch.h5")
