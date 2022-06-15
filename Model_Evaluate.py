from tensorflow.keras.models import load_model
import os
import tensorflow as tf



base_dir = 'G:/내 드라이브/Face Mask Dataset'
test_dir = os.path.join(base_dir, 'Test')

image_size = 224
batch_size = 32



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size = (image_size, image_size),
                                             batch_size = batch_size,
                                             class_mode = 'categorical')

model = load_model('MD.h5')

scores = model.evaluate(test_data, steps=30)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))