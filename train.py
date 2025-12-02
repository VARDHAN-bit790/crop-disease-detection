import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Paths and Configs
DATASET_PATH = "PlantVillage"
MODEL_PATH = "model/crop_disease_model.tf.keras"
IMGSIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30  # Increased for better learning

os.makedirs("model", exist_ok=True)

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
)

train_dataset = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMGSIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_dataset = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMGSIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

classnames = list(train_dataset.class_indices.keys())
print("Classes found:", classnames)

# Build CNN with BatchNorm and Dropout
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMGSIZE[0], IMGSIZE[1], 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(classnames), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training with EarlyStopping and Checkpoint
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_PATH)
print(f"Model trained and saved at {MODEL_PATH}")
