from .ImageSimilarity import getSimilarImages
from .generate_embeddings import generateEmbeddings

def generate_embeddings(model_name, img_directory):
    generateEmbeddings(model_name, img_directory)

def get_similar_images(model_name, query_image_path, img_directory, topK):
    return getSimilarImages(model_name, query_image_path, img_directory, topK)