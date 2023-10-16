# Hashing_for_geo-localization
This contains the codes and usage method described in "Hashing for Geo-localization", TGRS 2023.

![image](https://github.com/TaoyiminR/Hashing_for_geo-localization/blob/main/Framework.png)

# Abstract
In this paper, we undertake the task of fast geolocalization of a query ground image by using geo-tagged aerial images. To this end, we propose a hashing strategy that fast searches the database of geo-tagged aerial images for the ground image’s matches, whose geo-tags are exploited to estimate the ground geographic location. Speciﬁcally, we commence by converting the aerial images into ground-view aerial images that have the common angle of view (i.e., horizontal view) with the ground image. We then develop a feature extraction model and a hash encoder for generating hash codes for the images. Based on these models, the ground image and the geo-tagged aerial images are transformed to hash codes that comprehensively reﬂect their visual content similarity. Fast searching the geo-tagged aerial image database for the ground image’s matches is conducted subject to small Hamming distance between the hash codes. We extract a geographical cluster from the matched aerial images subject to their geo-tags. In this way, the geographic location of the ground image is efﬁciently retrieved according to the geographical cluster. Experiments on two datasets validate the efﬁciency and effectiveness of our proposed framework. We have released our implementation code at https://github.com/taoyiminR/Hashing_for_geo-localization for public evaluation.

# Datasets
We adopt two datasets to verify our method, including CVUSA and CVACT dataset.    
  CVUSA dataset: The dataset can be accessed from https://github.com/viibridges/crossnet  
  CVACT dataset: The dataset can be accessed from https://github.com/Liumouliu/OriCNN  

# Codes
You can use the train.py to train the feature extraction model and use the test.py to test this model.  
After obtain the feature descriptors of the image, you can use the encoding.py to train and encode the image.  
The file location_clustering.py is used to geographically cluster based localization.
