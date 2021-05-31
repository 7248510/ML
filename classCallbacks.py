#Training Fashion MNIST
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Setting the log level
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np 
class accuracyCallback(tf.keras.callbacks.Callback):
    #logs = output from
    #classifications = model.predict(test_images) #If the accuracy rating > 0.95
    def on_epoch_end(self, epock, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nModel Reached 95% accuracy stopping training.")
            self.model.stop_training = True
callbacks = accuracyCallback()
data = tf.keras.datasets.fashion_mnist #This will grab the dataset from tensorflow. It caches.
(training_images, training_labels), (test_images, test_labels) = data.load_data() #loading tensorflows dataset
training_images = training_images / 255.0 #Gray scale
test_images = test_images / 255.0 #Gray scale
model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #layers
    keras.layers.Dense(128, activation=tf.nn.relu), #128 neurons
    keras.layers.Dense(10,activation=tf.nn.softmax) # 10 neurons, probability softmax finds the highest value
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])