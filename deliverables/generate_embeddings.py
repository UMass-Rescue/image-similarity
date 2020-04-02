import time
import image_similarity
import argparse
import os

def generateEmbeddings(model_name, img_directory, mydb):
    start_time = time.time()
    image_similarity.loadData(model_name, img_directory, mydb)
    print(time.time() - start_time)

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model_path", required=True,
    help="Absolute path to model which you want to use to generate embeddings")
ap.add_argument("-ip", "--input_path", required=True,
    help="Absolute Path to the directory of images for which you want to generate embeddings for similarity search")
ap.add_argument("-dp", "--database_path", required=True,
    help="Absolute Path to the sqlite database where you want to store the image embeddings")

args = vars(ap.parse_args())

if not os.path.exists(args['model_path']):
    print ("The model to generate embeddings does not exist.")
    exit()
if not os.path.exists(args['input_path']):
    print ("The input directory does not exist.")
    exit()
    
generateEmbeddings(args['model_path'], args['input_path'], args['database_path'])