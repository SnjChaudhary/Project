import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.makedirs('models', exist_ok=True)

cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    r'C:\Users\snjch\Desktop\predictive_maintenance_quality_control\data\product_images',  
    target_size=(64, 64), 
    batch_size=32,
    class_mode='binary'  
)

cnn_model.fit(train_generator, epochs=10)

cnn_model.save(r'C:\Users\snjch\Desktop\predictive_maintenance_quality_control\models\cnn_model.h5')

test_loss, test_acc = cnn_model.evaluate(train_generator)
print(f"Quality Control Model Test Accuracy: {test_acc * 100:.2f}%")
