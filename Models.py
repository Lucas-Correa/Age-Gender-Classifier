import keras
import tensorflow as tf
from keras import optimizers
from keras import regularizers
from keras.layers import Dense,Dropout,Conv2D, Flatten, MaxPooling2D,BatchNormalization
from keras import Sequential 

def Model_G1():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 3)))

    model.add(Conv2D(16, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 3))) 

    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 3))) 

    model.add(Flatten())

    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dense(units = 16, activation = 'relu'))
    model.add(Dense(units = 3, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
    
    return model


def Model_G2():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
  
    
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    
    model.add(Conv2D(256, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dense(units = 3, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
    
    return model

def Model_G3():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu', kernel_regularizer=regularizers.L2(l2= 0.002)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer=regularizers.L2(l2= 0.002)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer=regularizers.L2(l2= 0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer=regularizers.L2(l2= 0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(256, (3, 3), activation = 'relu', kernel_regularizer=regularizers.L2(l2= 0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Dense(units = 32, activation = 'relu'))

    model.add(Dense(units = 3, activation = 'softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
    
    return model

def Model_A1():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 3)))
    
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (3, 3)))
    
    model.add(Flatten())
    
    model.add(Dense(units = 128, activation = 'relu'))
        
    model.add(Dense(units = 32, activation = 'relu'))
    
    model.add(Dense(units = 7, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = tf.optimizers.RMSprop(learning_rate=0.0001), metrics=['acc'])
    
    return model

def Model_A2():
    model = Sequential()
    model.add(Conv2D(32, (1, 1), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (2, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(128, (2, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(256, (2, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))


    model.add(Flatten())


    model.add(Dense(units = 128, activation = 'relu'))


    model.add(Dense(units = 128, activation = 'relu'))

    model.add(Dense(units = 64, activation = 'relu'))

    model.add(Dense(units = 32, activation = 'relu'))

    model.add(Dense(units = 7, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = tf.optimizers.RMSprop(learning_rate=0.001), metrics=['acc'])

    return model

def Model_A3():
    model = Sequential()
    model.add(Conv2D(32, (1, 1), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(64, (2, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (2, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(256, (2, 2), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(512, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    model.add(Dense(units = 128, activation = 'relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(units = 128, activation = 'relu'))
    
    model.add(Dense(units = 64, activation = 'relu'))
        
    model.add(Dense(units = 32, activation = 'relu'))
    
    model.add(Dense(units = 7, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = tf.optimizers.RMSprop(learning_rate=0.001), metrics=['acc'])
    
    return model
