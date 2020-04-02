from .SceneClassification import get_image_scene
from .Embedding import getEmbedding
import os
import json
import heapq
import sqlite3

def getSimilarImages(model_name, query_image_path, img_directory, mydb, K):
    
    try:
        scene_type = get_image_scene(query_image_path)
    except:
        scene_type = ''
    
    conn = sqlite3.connect(mydb)
    distance = {}
    sql = "SELECT * FROM metadata where scene_type = (?)"
    val = (scene_type, )
    cursor = conn.execute(sql,val)
    resultset = cursor.fetchall()
    embedding_dict = dict()
    isEmbeddingPresent = False
    for result in resultset:
        imagePath = result[1]
        if imagePath == query_image_path:
            isEmbeddingPresent = True
        embedStr = result[2]
        embedding_dict[imagePath] = json.loads(embedStr)
    
    if not isEmbeddingPresent:
        query_image_embedding = getEmbedding(model_name, query_image_path).tolist()
    else:
        query_image_embedding = embedding_dict[query_image_path]
        del embedding_dict[query_image_path]

    for img in embedding_dict:
        candidate_embedding = embedding_dict[img]
        distance[img] = sum([(query_image_embedding[idx] - candidate_embedding[idx])**2 for idx in range(len(query_image_embedding))])**(0.5)
    
    heap = [(value, key) for key,value in distance.items()]
    largestK = heapq.nsmallest(K, heap)
    conn.commit()
    conn.close()
    return largestK