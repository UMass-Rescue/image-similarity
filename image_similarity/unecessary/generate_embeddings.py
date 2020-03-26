import numpy as np
from Embedding import getEmbedding, load_data
import os
import argparse
import json
import gc
from keras.preprocessing.image import load_img, img_to_array

def generateEmbeddings(model_name, img_directory):
    
#     img_embedding = dict()
#     cur_subdir = None
#     for subdir, dirs, files in os.walk(img_directory):
#         for filename in files:
#             if filename.endswith(".jpg") or filename.endswith(".png"):
#                 img_path = img_path = os.path.join(subdir, filename)
#                 if subdir != cur_subdir and cur_subdir != None:
#                         embedding_path = os.path.join(cur_subdir,"embeddings.json")
#                         with open(embedding_path,'w') as f:
#                             j = json.dumps(img_embedding)
#                             img_embedding.clear()
#                             f.write(j)
#                             f.close()
#                             gc.collect()
#                             print("Done for ", subdir)
#                 cur_subdir = subdir
#                 img_embedding[filename] = getEmbedding(model_name, img_path).tolist()
#             else:
#                 continue

#     img_embedding = dict()
#     c = 1
#     for filename in os.listdir(img_directory):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             img_path = os.path.join(img_directory, filename)
#             print(img_path)
#             img_embedding[filename] = getEmbedding(model_name, img_path).tolist()
#             c += 1
#             if c >= 2:
#                 break
#     with open("embeddings.json",'w') as f:
#         j = json.dumps(img_embedding)
#         f.write(j)
#         f.close()
#     load_data(model_name,img_directory)
            
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model_path", required=True,
#     help="Absolute path to model which you want to use to generate embeddings")
# ap.add_argument("-ip", "--input_path", required=True,
#     help="Absolute Path to the images for which you want to generate embeddings")
# args = vars(ap.parse_args())
# if not os.path.exists(args['model_path']):
#     print ("The model to generate embeddings does not exist.")
#     exit()
# if not os.path.exists(args['input_path']):
#     print ("The input directory does not exist.")
#     exit()
# generateEmbeddings(args['model_path'], args['input_path'])