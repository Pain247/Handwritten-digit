import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
seed = 7
numpy.random.seed(seed)
#load data from data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape

#Calculate pixels of images
num_pixels = X_train.shape[1] * X_train.shape[2] # 28x28
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
#Take input to [0,1]
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print y_test.shape[1]

#Baseline_model function
def baseline_model():
	#model Sequential
	model = Sequential()
	#Input layer
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
#Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print scores
predictions = model.predict(X_train)
print predictions
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print("Baseline Accuracy: %.2f%%" % (scores[1]*100))

