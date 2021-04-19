# Misplaced-Product-Identification-using-various-SOTA-Classification-models

This project integrates an anchor-free object detector (FCOS) with a multi-level feature pyramid network, which contains richer semantic features for object detection.

## Background

## Product Recognition is a Challenging Problem :
- Very similar products, yet entirely different
The same product can look very different under different conditions (orientation/packaging etc.) 
Our work has focus historically on such challenges for faces and objects in general.

![image](https://user-images.githubusercontent.com/54212099/115286524-29107500-a11d-11eb-9b09-e7527f99efdd.png)![image](https://user-images.githubusercontent.com/54212099/115286554-30d01980-a11d-11eb-9d16-bc15b80e5c0b.png)

- Packaging can make the same product look very different.
For the purposes of inventory management, they are to be counted as the original product
![image](https://user-images.githubusercontent.com/54212099/115286353-fa929a00-a11c-11eb-879e-a51feceb6438.png)

- Orientation presents one of the biggest challenges
The same product looks very different from different angles
![image](https://user-images.githubusercontent.com/54212099/115286661-4ba28e00-a11d-11eb-99f1-86ee13d60f83.png)

## Dataset

The dataset is a private dataset provided by Bossonova Robotics, and consists of over 5 million images of panograms of every aisle across the store, for several Walmart stores across Pennsylvania

## Data Cleaning

Since the dataset is large, manual cleaning and annotation would take several man hours. In order to reduce that two algorithms were developed :
 - ## Clustering
    - The idea is to create clusters with different viewpoints (front, back, side) or packagings of the same product so that selecting top N images from each cluster will cover all different viewpoints of the product.
    - Calculated the mean cosine similarity for each cluster that was generated in order to identify clusters which had least distinct images among all clusters (Dump cluster).
    - Plotted elbow curve for each product to get the best value of K (number of clusters) for K-Means

![image](https://user-images.githubusercontent.com/54212099/115292866-8d830280-a124-11eb-8376-10aa9bf3301e.png)

- ## Autoencoder model
  - The idea was model to generate robust embedded features to identify the misplaced products by comparing its feature vector with the prototype of the actual (Target) product belonging to that ID using cosine similarity. Based on the similarity scores the product crop was assigned a respective folder

![image](https://user-images.githubusercontent.com/54212099/115293005-c622dc00-a124-11eb-9f9f-120b8112bb25.png)


## Data Collection by UPC Mapping :

Training a model on product category ( UPC – unique no associated to product type from the company its been manufactured by ) helps to deal with the problem of modeling all variations per category.
![image](https://user-images.githubusercontent.com/54212099/115287083-d97e7900-a11d-11eb-9117-23a3dd57cf8c.png)

## End to End Architecture 
- Given a Panogram image collected by a robot, after scanninng an aisle, we generate the crops for the all the products using Retinanet model with the Resnet50 Backbone and FPN as the neck
![image](https://user-images.githubusercontent.com/54212099/115289790-f9636c00-a120-11eb-8993-7b5c4c934174.png)

- The individual crops are collected for a particular UPC and trained. We find from several different asrchitectures that the B4-Efficient Model works best for our dataset. We train our model on eight NVIDIA's Titan RTX GPUs This training phase includes  :
   - Extracting the features for the top N images selected using the trained product recognition feature extractor block 
   - Enrolling these features along with their image stats into the product database with associated UPC code.
- For testing we run the product detection model on the pano and then associate products with their corresponding label based and then associate products to the upc from the labels.

![image](https://user-images.githubusercontent.com/54212099/115290803-f61cb000-a121-11eb-9818-40cdc05e30e4.png)

Now in testing further :

  - For a product, query the product database(created during training phase) to check if there any entries for its associated UPC.

  - If there are entries corresponding to the UPC, extract the feature for the product and compare it with all the queried features using a  product matching network.

  - A product is predicted to be a plug if it is not matched with any of the entries corresponding to its UPC in the database.

![image](https://user-images.githubusercontent.com/54212099/115291346-a094d300-a122-11eb-9934-731ad0c88463.png)

Details about various classification model experiments and hyperparameter tunings are mentioned in the presentation module.

## Acknowledgements
Our project is based on the following two github repositories:

https://github.com/open-mmlab/mmdetection

We would like to thank the contributors for providing us code resources

## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors.
