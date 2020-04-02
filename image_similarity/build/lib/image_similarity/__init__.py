from .ImageSimilarity import getSimilarImages
from .Embedding import loadData

def generate_embeddings(model_name, img_directory, mydb):
    loadData(model_name, img_directory, mydb)

def get_similar_images(model_name, query_image_path, img_directory, mydb, topK):
    return getSimilarImages(model_name, query_image_path, img_directory, mydb, topK)