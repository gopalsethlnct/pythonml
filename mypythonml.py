
from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.models import Sequential

model=Sequential()

model.add(Convolution2D(filters=32, 
                        kernel_size=(5,5), 
                        activation='relu',
                   input_shape=(200,200, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32, kernel_size=(5,5), 
                        activation='relu',
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(filters=32, 
                        kernel_size=(5,5), 
                        activation='relu',
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'myimagetrain/',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'myimagetest',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')
model.fit_generator(
        training_set,
        steps_per_epoch=100,
        epochs=1,
        validation_data=test_set,
        validation_steps=10)

model.save('myface.h5')

