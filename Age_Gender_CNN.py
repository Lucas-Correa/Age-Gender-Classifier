import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Dropout,Conv2D, Flatten, MaxPooling2D,BatchNormalization
from keras.models import Sequential 
from keras import optimizers

from Preprocessing import get_organize_files


def Age_Model():
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
    model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['acc'])
    
    return model
    

def Gender_Model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (5, 5), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(128, (7, 7), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(256, (7, 7), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 16, activation = 'relu'))
    model.add(Dense(units = 3, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
    
    return model

def viz_overfit(history,test_generator,model_dir):
    
    model = history.model
    epc = history_age.params['epochs']
    batch_size = history_age.params['steps']
    target_names = list(test_generator.class_indices.keys())
    num_of_test_samples = test_generator.batch_size

    acc_train = history.history['acc']
    acc_val = history.history['val_acc']
    epochs = range(1,epc+1)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(model_dir+'Training and Validation accuracy')

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,epc+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(model_dir+'Training and Validation loss')

    Y_pred = model.predict(test_generator, math.ceil(num_of_test_samples / batch_size))
    
    print(classification_report(test_generator.classes[:Y_pred.shape[0]],
                                list(np.argmax(Y_pred,axis=1)), 
                                target_names=target_names))


if __name__ == '__main__':

    tf.device('/cpu:0')

    model_age = Age_Model()

    train_generator,validation_generator, test_generator = get_organize_files()

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh_%Mm")
    model_dir = 'Models/Model_{}/'.format(timestamp)
    checkpoints = model_dir + '{epoch:02d}-{val_accuracy:.2f}.hdf5'

    callback_list = [
        keras.callbacks.EarlyStopping(monitor='acc',patience=7),
        keras.callbacks.ModelCheckpoint(filepath=checkpoints,save_best_only=True, monitor='acc'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,cooldown=2)
    ]
    history_age = model_age.fit(train_generator,
                            steps_per_epoch=93,
                            epochs=50,
                            callbacks=callback_list,
                            validation_data = validation_generator,
                            validation_steps=93)

    viz_overfit(history_age, test_generator,model_dir)