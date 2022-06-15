import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = 'G:/내 드라이브/Face Mask Dataset'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')

image_size = 224
target_size = (image_size, image_size)
input_shape = (image_size, image_size, 3)
batch_size = 32
epochs = 25

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                             shear_range = 0.2,
                                                             zoom_range = 0.2,
                                                             width_shift_range = 0.2,
                                                             height_shift_range = 0.2,
                                                             fill_mode="nearest")

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)



train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size = (image_size, image_size),
                                               batch_size = batch_size,
                                               class_mode = 'categorical')

validation_data = validation_datagen.flow_from_directory(validation_dir,
                                             target_size = (image_size, image_size),
                                             batch_size = batch_size,
                                             class_mode = 'categorical')

categories = list(train_data.class_indices.keys())
print(train_data.class_indices)

base_model = tf.keras.applications.MobileNet(weights = "imagenet",
                                             include_top = False,
                                             input_shape = input_shape)

base_model.trainable = False

inputs = keras.Input(shape = input_shape)

x = base_model(inputs, training = False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(len(categories), 
                          activation="sigmoid")(x)

model = keras.Model(inputs = inputs, 
                    outputs = x, 
                    name="maskDetection")

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer = optimizer,
              loss = tf.keras.losses.BinaryCrossentropy(from_logits = False),
              metrics=[keras.metrics.BinaryAccuracy(), 
                       'accuracy'])

history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=epochs,
                    steps_per_epoch=150,
                    validation_steps=100)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,loss,c="red",label="Training")
plt.plot(epochs,val_loss,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()




acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

#Save model
model.save('MD.h5')