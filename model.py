import csv
import cv2
from os import path
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

    
#load csv file
def loadData(basePath):

    lines = []
    with open(path.join(basePath,'driving_log.csv')) as f:
        content = csv.reader(f)
        for line in content:
            lines.append(line)

    return lines

#brightness imagens
def brightness_change(image):

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.2,0.8)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1
	
	
#generator to yeild processed images for training as well as validation data set
def generator(data, batchSize = 32):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), int(batchSize/4)):
            X_batch = []
            y_batch = []
            details = data[i: i+int(batchSize/4)]
            for line in details:

                image = brightness_change(mpimg.imread('./data/IMG/'+ line[0].split('/')[-1]))
                steering_angle = float(line[3])

                #appending original image
                X_batch.append(image)
                y_batch.append(steering_angle)

                #appending flipped image
                X_batch.append(np.fliplr(image))
                y_batch.append(-steering_angle)

                # appending left camera image and steering angle with offset
                X_batch.append(brightness_change(mpimg.imread('./data/IMG/'+ line[1].split('/')[-1])))
                y_batch.append(steering_angle+0.2)

                # appending right camera image and steering angle with offset
                X_batch.append(brightness_change(mpimg.imread('./data/IMG/'+ line[2].split('/')[-1])))
                y_batch.append(steering_angle-0.2)
				
            # converting to numpy array
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield shuffle(X_batch, y_batch)
			
#creating model to be trained
def network_model():

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(32,3,3,activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,3,3,activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D())    
    model.add(Conv2D(128,3,3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256,3,3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(20))
    model.add(Dense(1))
	
    return model


# load the csv file
basePath = 'C:/Users/ednaldo.goncalves/Documents/GitHub/CarND-BehavioralCloning-P4/data/'
print('loading the data...')
samples = loadData(basePath)

#Diving data among training and validation set
train_samples, validation_samples  = train_test_split(samples, test_size = 0.2)

# define the network model
model = network_model()
model.summary()

# model compile		
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(generator(train_samples), samples_per_epoch = len(train_samples), nb_epoch = 4, validation_data=generator(validation_samples), nb_val_samples=len(validation_samples), verbose=1)

# save model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
