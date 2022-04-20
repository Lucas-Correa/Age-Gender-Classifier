import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import tarfile
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator

def file_organizer(images_split,folder_name):
    #creates directories for training, test and validation files
    cwd = os.getcwd()
    
    split_folder = os.path.join(cwd,folder_name)
    if not os.path.exists(split_folder):
        os.mkdir(split_folder)

    #copy images to each directory
    for index, row in images_split.iterrows():
        src = os.path.join(cwd,'aligned',row['user_id'],'landmark_aligned_face.'+str(row['face_id'])+'.'+row['original_image'])
        dst = os.path.join(split_folder,row['original_image'])
        shutil.copyfile(src,dst)

    print('{} folder created'.format(folder_name))

    return split_folder

def get_organize_files(): 
    #download files with images
    #https://talhassner.github.io/home/projects/Adience/Adience-data.html
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
     
    train = fold.groupby('gender',as_index=False,group_keys=False).apply(lambda x: x.sample(frac=.6))
    test = fold.drop(train.index).groupby('gender',as_index=False,group_keys=False).apply(lambda x: x.sample(frac=.4))
    validation = fold.drop(train.index).drop(test.index)
    
    print('Train, Validation and Test split done')

    return train,validation,test

def ImgGen(df,col,dir):

    if 'train' in dir:
        datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=45,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    fill_mode='nearest')
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_dataframe(
                    df,
                    x_col='original_image',
                    y_col=col,
                    directory=file_organizer(df,dir),
                    target_size=(120,120),
                    batch_size=100,
                    class_mode='categorical'
                )
    return generator