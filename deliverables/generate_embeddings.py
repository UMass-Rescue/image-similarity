import time
import image_similarity
import argparse
import os

def generateEmbeddings(model_name, img_directory, mydb):
    start_time = time.time()
    image_similarity.loadData(model_name, img_directory, mydb)
    print(time.time() - start_time)

file = open("path_setup.txt", "r")
path_dict = dict()
for line in file:
    line = line.rstrip('\n')
    path_str = line.split('=')
    path_dict[path_str[0]] = path_str[1]
print(path_dict)
db_path = path_dict['db_path']
model_path = path_dict['model_path']
img_directory = path_dict['img_directory']

generateEmbeddings(model_path, img_directory, db_path)