import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from datetime import datetime

import keras
import tensorflow as tf

from Preprocessing import get_organize_files, ImgGen
from Models import Model_A3, Model_G3


def viz_overfit(history,test_generator,model_dir):

    #inicialize some values for the graph
    model = history.model
    epc = history.params['epochs']
    batch_size = history.params['steps']
    target_names = list(test_generator.class_indices.keys())
    num_of_test_samples = test_generator.batch_size

    #train vs validation accuracy graph
    acc_train = history.history['acc']
    acc_val = history.history['val_acc']
    epochs = range(1,epc+1)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(model_dir+'Training_Validation_accuracy')
    plt.clf()

    #train vs validation loss graph
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,epc+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training_Validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_dir+'Training and Validation loss')
    plt.clf()

    #report with some metrics about a predction using the test set, unseen by the model
    Y_pred = model.predict(test_generator, math.ceil(num_of_test_samples / batch_size))
    
    print(classification_report(test_generator.classes[:Y_pred.shape[0]],
                                list(np.argmax(Y_pred,axis=1)), 
                                target_names=target_names))


if __name__ == '__main__':

    tf.device('/cpu:0')

    train, validation, test = get_organize_files()

    timestamp = datetime.now().strftime("%Y_%m_%d_%Hh_%Mm")

    ########################### AGE ###########################

    model_age = Model_A3()

    model_dir = 'Models/Model_{}/Age/'.format(timestamp)
    checkpoints = model_dir + '{epoch:02d}-{val_acc:.2f}.hdf5'
    
    callback_list = [
        keras.callbacks.EarlyStopping(monitor='acc',patience=7),
        keras.callbacks.ModelCheckpoint(filepath=checkpoints, monitor='val_acc'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,cooldown=2)
    ]
    history_age = model_age.fit(ImgGen(train,'age','train_images'),
                            epochs=20,
                            callbacks=callback_list,
                            validation_data = ImgGen(validation,'age','validation_images')
                            )
    
    viz_overfit(history_age, ImgGen(test,'age','test_images'),model_dir)


    ########################### GENDER ###########################

    model_gender = Model_G3()
    
    model_dir = 'Models/Model_{}/Gender/'.format(timestamp)
    checkpoints = model_dir + '{epoch:02d}-{val_acc:.2f}.hdf5'
    
    callback_list = [
        keras.callbacks.EarlyStopping(monitor='acc',patience=7),
        keras.callbacks.ModelCheckpoint(filepath=checkpoints, monitor='val_acc')
       , #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,cooldown=2)
    ]

    
    history_gender = model_gender.fit(ImgGen(train,'gender','train_images'),
                        epochs=50,
                        callbacks=callback_list,
                        validation_data = ImgGen(validation,'gender','validation_images')
                        )
    
    viz_overfit(history_gender, ImgGen(test,'gender','test_images'),model_dir)