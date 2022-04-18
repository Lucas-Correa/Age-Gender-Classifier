import pandas as pd
import numpy as np
import datetime
import requests
from requests.auth import HTTPBasicAuth
import tarfile
import os
import shutil
import warnings
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Model,Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_organize_files(): 
    #download files with images
    #https://talhassner.github.io/home/projects/Adience/Adience-data.html
    BASE_URL = "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification"
    DOWNLOAD_FILES = {
              "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/aligned.tar.gz": "aligned.tar.gz",
              "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_0_data.txt": "fold_0_data.txt",
              "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_1_data.txt": "fold_1_data.txt",
              "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_2_data.txt": "fold_2_data.txt",
              "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_3_data.txt": "fold_3_data.txt",
              "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_4_data.txt": "fold_4_data.txt"
                    }

    for DOWNLOAD_FILE, FILE_NAME in DOWNLOAD_FILES.items():
        if not os.path.exists(FILE_NAME):
            print('starting download')
            with open(FILE_NAME, 'wb') as file:
                r = requests.get(DOWNLOAD_FILE, auth = HTTPBasicAuth('adiencedb', 'adience'))
                file.write(r.content)
                print('downloaded {}'.format(FILE_NAME))
    
    
    
    if not os.path.exists('aligned'):
        with tarfile.open('aligned.tar.gz') as file:
            print('unziping images...')
            file.extractall()
            print('images unziped')
        
   
    
    #compile all files catalog
    fold = pd.read_csv('fold_0_data.txt',sep='\t')
    fold.rename(columns={' user_id':'user_id'},inplace=True)
    fold['fold'] = 0
    for i in range(1,5):
        temp = pd.read_csv('fold_'+str(i)+'_data.txt',sep='\t')
        temp['fold'] = i
        fold = fold.append(temp,ignore_index=True)

    fold.dropna(subset=['gender'],inplace=True)
    fold = fold[['user_id','original_image','face_id','age','gender','fold']]
    
    fold = fold.loc[fold.age !='None']
    fold.age.replace(['35'     ,'13'    ,'22'     ,'34'     ,'45'      ,'(27, 32)','23'      ,'55'      ,'36'     ,'(38, 42)','57'      ,'58'      ,'46'      ,'3'      ,'29'     ,'2'     ,'42'],
                     ['(35,43)','(8,13)','(15,24)','(25,34)','(45,100)','(25,34)' ,'(15,24)','(45,100)','(35,43)','(35,43)' ,'(45,100)','(45,100)','(45,100)','(3, 6)','(25,34)','(0, 2)','(35,43)'],
                     inplace = True)
    fold.age.replace(['(38, 43)','(8, 12)','(15, 20)','(60, 100)','(38, 43)','(48, 53)','(4, 6)','(38, 48)','(25, 32)','(8, 23)'],
                     ['(35,43)' ,'(8,13)' ,'(15,24)' ,'(45,100)' ,'(35,43)' ,'(45,100)','(3, 6)','(45,100)','(25,34)' ,'(8,13)'],
                     inplace=True)
     
    train = fold.groupby('gender',as_index=False,group_keys=False).apply(lambda x: x.sample(frac=.5))
    test = fold.drop(train.index.values)
    
    print('Train and test split done')
    
    #creates directories for training, test and validation files
    cwd = os.getcwd()
    
    train_faces = os.path.join(cwd,'train')
    if not os.path.exists(train_faces):
        os.mkdir(train_faces)

    test_faces = os.path.join(cwd,'test')
    if not os.path.exists(test_faces):
        os.mkdir(test_faces)

    #copy images to each directory
    for index, row in train.iterrows():
        src = os.path.join(cwd,'aligned',row['user_id'],'landmark_aligned_face.'+str(row['face_id'])+'.'+row['original_image'])
        dst = os.path.join(train_faces,row['original_image'])
        shutil.copyfile(src,dst)

    print('train images copied')

    for index, row in test.iterrows():
        src = os.path.join(cwd,'aligned',row['user_id'],'landmark_aligned_face.'+str(row['face_id'])+'.'+row['original_image'])
        dst = os.path.join(test_faces,row['original_image'])
        shutil.copyfile(src,dst)

    print('test images copied')
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=45,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.1,
                                  zoom_range=0.1,
                                  fill_mode='nearest')

    train_generator = train_datagen.flow_from_dataframe(
                        train_images_ids,
                        x_col='original_image',
                        y_col='age',
                        directory=train_faces,
                        target_size=(150,150),
                        batch_size=100,
                        class_mode='categorical'
                    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
                        test_images_ids,
                        x_col='original_image',
                        y_col='age',
                        directory=test_faces,
                        target_size=(150,150),
                        batch_size=100,
                        class_mode='categorical',
                        shuffle=False
                    )
    return train_generator, test_generator




    
