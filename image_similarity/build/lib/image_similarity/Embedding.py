import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
from keras.applications.vgg16 import VGG16
from keras.backend import l2_normalize
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
import time
import os
import argparse
import json
import gc
from .SceneClassification import get_image_scene
import sqlite3
from tqdm import tqdm

def loadData(model_name, img_directory, mydb):
    
    logFile = open("output.log",'a+')
    logFile.write("\n*************START**************\n")
    start_time = time.time()
    
    images = [os.path.join(img_directory,name) for name in os.listdir(img_directory) if os.path.isfile(os.path.join(img_directory, name)) and (name.endswith(".jpg") or name.endswith(".png"))]
    N = len(images)
    batch_size = N
    canAlloc = False
    conn = sqlite3.connect(mydb)
    while not canAlloc:    
        try:
            data = np.empty((batch_size, 224, 224, 3), dtype=np.float64)
            if N % batch_size == 0:
                batch_len = N // batch_size
            else:
                batch_len = N // batch_size + 1
            image_index = 0
            # iterate over batches
            logFile.write("\nMaximally allocated batch size is %d\n" % batch_size)
            logFile.write("\nBatch length is %d\n" % batch_len)
            for i in range(batch_len):
                if i == batch_len - 1:
                    batch_size = N - image_index
                    if batch_size == 0:
                        break
                data = np.empty((batch_size, 224, 224, 3), dtype=np.float64)
                cur_image_index = image_index
                # iterate over images in a single batch
                print("Iteration ", i + 1, " / ", batch_len)
                print("*******************************")
                print("Embedding generation forward pass")
                itr_str = "\nIteration " + str(i+1) + " / " + str(batch_len) + "\n"
                logFile.write(itr_str)
                logFile.write("\n*******************************\n")
                logFile.write("\nA. Embedding generation forward pass\n")
                preprocess_start_time = time.time()
                for j in tqdm(range(batch_size)):
                    img_path = images[image_index]
                    image_arr = load_img(img_path)
                    image_arr = img_to_array(image_arr).astype("float64")
                    image_arr = transform.resize(image_arr, (224, 224))
                    image_arr *= 1. / 255
                    data[j,:,:,:] = image_arr
                    image_index += 1
                preprocessing_time = time.time() - preprocess_start_time
                logFile.write("\n 1. Pre-processing time is %f\n" % preprocessing_time)
                embedding_start_time = time.time()
                embeddings = generateEmbeddings(model_name, data).tolist()
                logFile.write("\n 2. Embedding generation time is %f\n" % (time.time() - embedding_start_time))
                values = []
                # prepare values for insert query
                print("*******************************")
                print("Storing embedding in the DB")
                logFile.write("\n*******************************\n")
                logFile.write("\nB. Getting scene type information\n")
                index = 0
                scene_start_time = time.time()
                for embedding in tqdm(embeddings):
                    embedStr = json.dumps(embedding)
                    imagePath = images[cur_image_index + index]
                    scene_type = ''
                    try:
                        scene_type = get_image_scene(imagePath)
                    except:
                        pass
                    val = (imagePath, embedStr, scene_type)
                    values.append(val)
                    index += 1
                total_scene_parse_time = time.time() - scene_start_time
                logFile.write("\n 1. Total time taken for scene parsing is %f\n" % total_scene_parse_time)
                logFile.write("\n*******************************\n")
                logFile.write("\nC. Storing embedding in the DB\n")
                store_start_time = time.time()
                sql = "INSERT INTO metadata (image_path, embedding, scene_type) values (?,?,?)"
                total_store_time = time.time() - store_start_time
                logFile.write("\n 1. Data insertion time is %f\n" % total_store_time)
                conn.executemany(sql, values)
                conn.commit()
                print("*******************************")
                logFile.write("\n*******************************\n")
            canAlloc = True
        except MemoryError:
            batch_size = int(batch_size/2)
    logFile.write("\nOverall generation time was %f\n" % (time.time() - start_time))
    conn.close()
            
def convnet_model_():
    vgg_model = VGG16(weights=None, include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda  x_: l2_normalize(x,axis=1))(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def generateEmbeddings(model_name, data):
    
    convnet_model = convnet_model_()
    first_input = Input(shape=(224,224,3))
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: l2_normalize(x,axis=1))(emb)

    model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    # model = deep_rank_model()

    model.load_weights(model_name)
    
    return model.predict([data, data, data])

def getEmbedding(model_name, image_name):
    
    convnet_model = convnet_model_()
    first_input = Input(shape=(224,224,3))
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: l2_normalize(x,axis=1))(emb)

    model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    model.load_weights(model_name)

    image1 = load_img(image_name)
    image1 = img_to_array(image1).astype("float64")
    image1 = transform.resize(image1, (224, 224))
    image1 *= 1. / 255
    image1 = np.expand_dims(image1, axis = 0)
    embedding1 = model.predict([image1, image1, image1])[0]
    return embedding1