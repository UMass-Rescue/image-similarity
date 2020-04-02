import sqlite3

conn = sqlite3.connect('image_similarity.db')
query = '''
CREATE TABLE metadata (
    imageId INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    image_path TEXT,
    embedding TEXT,
    scene_type TEXT
);
'''
conn.execute(query)
conn.close()