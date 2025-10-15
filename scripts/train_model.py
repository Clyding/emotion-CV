# train_model.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# Load dataset
# ==============================
img_size = (48, 48)
batch_size = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
def scale_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = train_ds.map(scale_img).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.map(scale_img).cache().prefetch(buffer_size=AUTOTUNE)

# ==============================
# Build model
# ==============================
def build_small_cnn(input_shape=(48,48,1), n_classes=7):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    return models.Model(inp, out)

model = build_small_cnn(input_shape=(48,48,1), n_classes=len(class_names))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==============================
# Train
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

os.makedirs("models", exist_ok=True)
model.save("models/fer_cnn.h5")
print("✅ Model saved at models/fer_cnn.h5")
