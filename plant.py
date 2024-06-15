import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Veri setinin yolu
data_dir = 'flowers'

# Resim boyutları ve batch size
img_height, img_width = 150, 150
batch_size = 32

# ImageDataGenerator ile resimleri yükleme ve ön işleme
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
# CNN modelinin oluşturulması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 çiçek türü olduğu için
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modelin özeti
model.summary()
# Modelin eğitilmesi
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size,
    validation_data=validation_generator,
    epochs=epochs
)

# Eğitim ve doğrulama kayıplarını ve doğruluklarını grafikte gösterme
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

from tensorflow.keras.preprocessing import image


def predict_flower_type(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_labels = list(train_generator.class_indices.keys())

    return class_labels[predicted_class]

def get_care_instructions(flower_type):
    care_data = pd.read_csv('flower_care_tr.csv')
    care_info = care_data[care_data['Çiçek Türü'] == flower_type].to_dict('records')
    if care_info:
        return care_info[0]
    else:
        return "Bilinmeyen Çiçek Türü"

# Eğitim sonraında örnek kullanım
flower_image_path = 'rose1.jpg'  # Çiçek resmi dosya yolunu burada veriyorum ancak burayı input olarak da alabilirsiniz.
predicted_flower_type = predict_flower_type(flower_image_path)
print(f"Tahmin Edilen Çiçek Türü: {predicted_flower_type}")

flower_type = predicted_flower_type  # Tanımlanan çiçek türü
care_instructions = get_care_instructions(flower_type)
print(care_instructions)
