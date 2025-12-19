import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

test_dir = r"C:\Users\Wintel\Downloads\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\test"

model = tf.keras.models.load_model("best_fish_model.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

loss, acc = model.evaluate(test_gen)
print("Test Accuracy:", acc)

y_pred = model.predict(test_gen)
y_pred_classes = y_pred.argmax(axis=1)

print(classification_report(test_gen.classes, y_pred_classes))
