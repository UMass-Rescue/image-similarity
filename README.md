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

**Week of 3/12/20 - 3/19/20 : 7**

**Week of 3/19/20 - 3/26/20 : 8**

**Week of 3/26/20 - 4/2/20 : 9**

**Week of 4/2/20 - 4/9/20 : 10**

**Week of 4/9/20 - 4/16/20: 11**

**Week of 4/16/20 - 4/23/20: 12**

**Week of 4/23/20 - 4/30/20: 13**
