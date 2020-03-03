from SceneClassification import get_image_scene
from Embedding import getEmbedding
import os
import json
import heapq

def getSimilarImages(model_name, query_image_path, img_directory, K):
    
    query_image_embedding = getEmbedding(model_name, query_image_path).tolist()    
    class_dir = get_image_scene(query_image_path)
    srch_dir = os.path.join(img_directory, class_dir)
    embedding_file_path = os.path.join(srch_dir, "embeddings.json")
    with open(embedding_file_path,'r') as f:
        embedding_dict = json.load(f)
    distance = {}
    for img in embedding_dict:
        candidate_embedding = embedding_dict[img]
        distance[img] = sum([(query_image_embedding[idx] - candidate_embedding[idx])**2 for idx in range(len(query_image_embedding))])**(0.5)
    heap = [(value, key) for key,value in distance.items()]
    largestK = heapq.nsmallest(K, heap)
    largestK = [os.path.join(srch_dir, key) for value, key in largestK]
    return largestK