import argparse
import logging
import os
import sys

from keras import Sequential, Input, Model
from keras.src.applications import MobileNet
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from keras.src.optimizers import Adam
from keras.src.preprocessing.image import ImageDataGenerator

from app.utils.download_utils import download_and_unzip


logging.basicConfig(stream=sys.stdout,
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO
                    )


def build_model(img_size):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Large dense layer
    predictions = Dense(1, activation='sigmoid')(x)  #

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-path', type=str, default='../../output/models')
    parser.add_argument('--data-path', type=str, default='../../output/data')
    args = parser.parse_args()

    os.makedirs(args.models_path, exist_ok=True)

    dataset_name = "cats_and_dogs_filtered"
    data_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    training_dir = os.path.join(args.data_path, dataset_name, "train")
    validation_dir = os.path.join(args.data_path, dataset_name, "validation")
    batch_size = 32
    image_size = (224, 224)
    num_classes = 2  # Cats and Dogs
    epochs = 20

    # Download and unzip the dataset
    base_dir = os.path.dirname(training_dir)
    if not os.path.exists(base_dir):
        download_and_unzip(data_url, args.data_path)

    # Prepare image data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(base_dir, 'validation'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    model = build_model(image_size)

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # Add checkpoints and early stopping
    callbacks = [
        ModelCheckpoint(os.path.join(args.models_path, "model_best.h5"), save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    ]

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        callbacks=callbacks
    )

    logging.info("Model trained and saved successfully.")

    # Optionally load and predict
    # from tensorflow.keras.models import load_model
    # model = load_model('cats_vs_dogs_model.h5')
    # # Prediction code can be added here

