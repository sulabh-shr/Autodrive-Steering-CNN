import csv
import cv2
import numpy as np
from time import time

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

'''
    FUNCTION TO LOAD THE CSV FILE
        iterating through the elements (lines) of csv reader
        appending to the list: lines
        returning the list of lines
'''


def load_csv_file(path_to_csv):
    lines = []
    with open(path_to_csv) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines


def load_training_data_from_csv_lines(csv_lines):
    images = []  # List to store images
    measurements = []  # List to store corresponding steering measurement
    correction = 0.2  # Correction factor for non-center images

    print(len(csv_lines))
    for line in csv_lines[1:]:  # 1st line is the header row
        '''
            STEERING DATA ARRAY
                Making a steering measurement data array
                1st element is the center steering
                2nd element is the left steering
                3rd element is the right steering
        '''
        steering = []
        steering_center = float(line[3])
        steering.append(steering_center)
        steering.append(steering_center + correction)  # steering_left
        steering.append(steering_center - correction)  # steering_right

        for i in range(3):  # iterating through center, left, right
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = 'udacity_data/data/IMG/' + filename
            image = cv2.imread(current_path)
            images.append(image)
            measurements.append(steering[i])
            '''
                AUGMENTING DATA
                    Flipping the image
                    Reversing the measurement data
            '''
            images.append(np.fliplr(image))
            measurements.append((-1.0) * steering[i])
    return np.array(images), np.array(measurements)


start = time()

csv_path = 'udacity_data/data/driving_log.csv'
lines_from_csv = load_csv_file(csv_path)
X_train, Y_train = load_training_data_from_csv_lines(lines_from_csv)


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
