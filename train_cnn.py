import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = r"C:\Users\Wintel\Downloads\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train"
val_dir   = r"C:\Users\Wintel\Downloads\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val"


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)


model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)


model.save("cnn_fish_model.h5")
print("✅ CNN Model Saved")
