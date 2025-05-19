import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# Preproccesing and loading of the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train,X_Test= X_train/255.0, X_test/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

#Defining the CNN model
model=models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), #Adding the convolutional layer with 32 filters, 3x3 kernel and with ReLu activation function
    layers.MaxPooling2D((2, 2)), #Adding the max pooling layer with 2x2 size
    layers.Conv2D(64, (3, 3), activation='relu'), #Adding the 2nd convolutional layer 
    layers.MaxPooling2D((2, 2)), #Adding the 2nd max pooling layer'
    layers.Conv2D(64, (3, 3), activation='relu'), #Adding the 3rd convolutional layer
    layers.Flatten(), #Flattening the output of the last convolutional layer
    layers.Dense(64, activation='relu'), # dense layer with 64 neurons and ReLu activation function
    layers.Dense(10, activation='softmax') #output layer with 10 neurons and softmax activation function
])

# Model compiling
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) 
model.summary() 
# Model training
history=model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test)) 
# Model evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

#Model prediction for first 5 images
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test[:5].flatten()

print("\nPredictions for first 5 test images:")
for i in range(5):
    print(f"Image {i+1}:")
    print(f"  Predicted class: {class_names[predicted_classes[i]]} (probability: {predictions[i][predicted_classes[i]]:.4f})")
    print(f"  True class: {class_names[true_classes[i]]}")
