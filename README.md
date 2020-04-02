# Image Similarity

Spring 2020

Kautilya Rajbhara 

## Goal

To build a deep-learning based image-similarity model, which given a query image returns a list of similar images.

# Sample Output

Query Image:- 

![im1](https://github.com/UMass-Rescue/image-similarity/blob/master/output-samples/mountain-A.jpg)

Results :

![im2](https://github.com/UMass-Rescue/image-similarity/blob/master/output-samples/mountain-B.jpg)
![im2](https://github.com/UMass-Rescue/image-similarity/blob/master/output-samples/mountain-D.jpg)
![im2](https://github.com/UMass-Rescue/image-similarity/blob/master/output-samples/mountain-E.jpg)


## Utility

While analysing images recovered from a crime scene, A law enforcement official may want to look at similiar images from the past to get important insights for his/her investigation.

## Project Setup

Follow the below mentioned steps to setup the project:
1. Clone the repository. After that, download the deep learning model used by the application from [here](https://drive.google.com/file/d/1TmUKqp_TnzSP0TeAHIyTv8jG4KZeNqQP/view) and store the model file in the deliverables folder of the cloned repository.
2. If you are not using Anaconda, please install the latest version of Anaconda. Although it is possible to setup this project without Anaconda, using it will make life easier for you. All the below instruction makes an assumption that you installed Anaconda.
3. Open the terminal / command prompt and run the following commands (They create a conda environment where we will install all the dependencies for our module): <br>
  **conda create -n cs-696-image_similarity python=3.6.10 anaconda** <br>
  **conda activate cs-696-image_similarity**
4. In the same terminal session navigate to the deliverables folder and run the command **chmod +x setup_env.sh** to give execute permission to the shell script. Run the shell script using **./setup_env.sh** command. This script installs all the dependencies as well as creates a SQLite database that will be used by the image-similarity module. Here, if using Windows, than run each line in the setup_env.sh sequentially one by one on the command prompt.
5. After setting up the environment run the following command in the terminal / command prompt to install the image-similarity module <br> 
**pip install image_similarity-0.1-py3-none-any.whl**
6. Once the image similarity module is installed, run the **generate_embeddings.py** script. This script generates embedding for all the images in a directory and stores in the SQLite database which in turn is used to answer similarity queries. This is a compute intensive task and it takes time. When I ran it on 36.5K images from Places365 dataset, it took me around 6 hours. Depending on your computing resource it may take proportionately more or less time for your dataset. Please note this is just a one time thing, after embedding for all the images are created we can directly use it to answer user queries. We plan to use a batch job to generate embedding for any image that is added to the directory at a later time, so as to keep the db updated and correctly answer queries. The command to run is as follows: <br>
**python generate_embeddings.py -m <absolute_path_to_model(deepranking-v2-150000.h5)> -ip <absolute_path_to_your_images_directory> -dp image_similarity.db**
7. After the execution of the above mentioned script we are all set to test the image-similarity tool. Open the **similar-images-retrieval.ipynb** using JUPYTER. In the notebook, **click on Kernel->Change kernel->Select Python (cs-696-image_similarity)**. Follow the steps in the notebook to run and test the image similarity module.I hope you find the above instructions helpful and are able to setup and run the image similarity module. In case you are struct somewhere or need any help with setting up the module please feel free to reach out to me.

* You can have a look at the SQLite database which I ran for Places365 dataset [here](https://drive.google.com/file/d/1hgRKrvxeddJWqxb7wW8zKxQwBvX6lH3C/view).


## Weekly Blog

**Week of 1/30/20 - 2/6/20 : 1**
 
- Decided to work on school uniform dataset, specifically to detect school logo and use that to classify to which school does that logo   belongs.
- Since, school uniform dataset was not yet available decided to train logo detector for brand logo detection task so that in future can   use it as a pre-trained model and fine tune it for school logo detection.
- Developed understanding about object detection tasks in general.
- Explored various datasets for the task of brand logo detection. Came across several such datasets like FlickrLogo-32, WebLogo-2M,       TopLogo-10, Flickr Logo 27, Logo32plus. 
- Training a logo detector using SSD architecture. 

**Week of 2/6/20 - 2/13/20 : 2**

- Explored various other architectures like ssd inception, yolo, faster_rcnn, R-FCN etc.
- Decided to proceed with using faster_rcnn for better results.
- Trained logo detector using faster_rcnn architecture, specifically used faster_rcnn model pre-trained on MS-COCO dataset.
- Initial results -> faster_rcnn better than ssd but training is very slow.
- Dan said that he will be able to provide the __school uniform dataset__ only in __early March__. So, after discussing this with Brian,   decided to switch to working on building an image similairty model.

**Week of 2/13/20 - 2/20/20 : 3**

- Explored various approaches for finding image similarity.
- Adopted an implementation of DeepRanking to find similarity between images.
- Built an image based query retrieval engine to get the top 5 best images. 

**Week of 2/20/20 - 2/27/20 : 4**

- Improved the results by including the knowledge about an image's scene from Vivek’s scene classification model.
- Search images only from the directory corresponding to the scene.
- Worked on improving the query performance for image retrieval by pre-calculating the embeddings for images and only computing distances during runtime.

**Week of 2/27/20 - 3/5/20 : 5**

- Restructured the code and made it modular, such that it can work with any of the model that I try in the future without much changes.
- Packaged my code and created a wheel file so that it can be shipped to Dan.


**Week of 3/5/20 - 3/12/20 : 6**
 
*PASS WEEK*

**Week of 3/12/20 - 3/19/20 : 7**

*SPRING BREAK*

**Week of 3/19/20 - 3/26/20 : 8**

- Implemented dynamic batching to generate embeddings.
- Implemented logic to store the image embeddings in MySQL database as well to retireve the top K similar images from the database to answer the search query.
- Ran the forward pass to generate embeddings of 36.5K images obtained from Places dataset, and stored in a MySQL database.


**Week of 3/26/20 - 4/2/20 : 9**

- As per Brian's suggestion changed the DB from MySQL to SQLite.
- Finalized the image-similarity module
- Created the documentation as well as built necessary scripts for Dan to install the module and run it on his local machine.

**Week of 4/2/20 - 4/9/20 : 10**

**Week of 4/9/20 - 4/16/20: 11**

**Week of 4/16/20 - 4/23/20: 12**

**Week of 4/23/20 - 4/30/20: 13**
