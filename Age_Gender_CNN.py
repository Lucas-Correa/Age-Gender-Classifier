import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Model,Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Preprocessing import download_clean_processImages

def Age_Model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(256, (1, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    #model.add(Conv2D(512, (1, 1), activation = 'relu'))
    #model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(units = 32, activation = 'relu'))
    
    model.add(Dense(units = 7, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    
    return model

def Gender_Model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (1, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 192, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dense(units = 3, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    
    return model

def viz_overfit(model,history,num_of_test_samples,batch_size,target_names,epc):
    
    Y_pred = model.predict_generator(test_generator, math.ceil(num_of_test_samples / batch_size))
    
    print(classification_report(test_generator.classes[:Y_pred.shape[0]],
                                list(np.argmax(Y_pred,axis=1)), 
                                target_names=target_names))


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
    

train_generator,test_generator, download_clean_processImages()

model_age = Age_Model()

history_age = model_age.fit(train_generator,
                          steps_per_epoch=93,
                          epochs=50,
                          validation_data = test_generator,
                          validation_steps=93)
model_age.save('face_age_detection.h5')

viz_overfit(model_age,history_age, 9276,100,list(train_generator.class_indices.keys()),25)