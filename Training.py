import numpy as np 
import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import Adam
import pickle
import os

img = load_img('./fruits-360/Training/Banana Lady Finger/0_100.jpg')
#image = cv2.imread('./fruits-360/Training/Banana Lady Finger/0_100.jpg')
#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(image.shape)

datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

img = np.array(img)
img = img.reshape((1,) + img.shape )

i=0
for batch in datagen.flow(img, batch_size = 1, 
            save_to_dir='fruits-360/Training', save_prefix='fruits', save_format='jpeg'):
            i += 1
            if i>20:
                break

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
        'fruits-360/Training',  
        target_size=(150, 150),  
        batch_size=batch_size,
        class_mode='binary')  

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'fruits-360/Test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=50 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('Fruits.h5')

print('Executed Successfully')