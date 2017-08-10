import csv
import cv2
import numpy as np
from time import time

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('udacity_data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:-1]
    # 1st line is header

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'udacity_data/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])  # Taking the steering angle as label
    measurements.append(measurement)
    '''
        AUGMENTING DATA
            Flipping the image
            Reversing the measurement data
    '''
    images.append(np.fliplr(image))
    measurements.append((-1.0) * measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

start = time()

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
#   dividing by max and subtracting by 0.5 to normalize around 0
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
#   cropping top 70 pixels and bottom 25 pixels and 0 left and 0 right pixels
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 6, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=1, verbose=1)

model.save('model.h5')

print('Elapsed time: ', time() - start)
