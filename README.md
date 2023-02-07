# still updating more details....
# SI699_Demo
Demo for SI699

## Table of Contents

- [Background](##Unsupervised-Aspect-Category-Detection)
- [Data](##Data)
    - [Dataset](##Dataset)
- [Dependencies](##Dependencies)
-  [Setup](##Setup:Install packages)
- [File introduction](##File introduction)
- [Result](##Result)

- [Cite](##Cite)


## Unsupervised-Aspect-Category-Detection
This repository contains the part of code for the paper "An Unsupervised Approach for Aspect Category Detection Using Soft Cosine Similarity Measure".

## Data
The unlabeld yelp reviews sentences can be downloaded at [[Download]](https://drive.google.com/file/d/1aCOK59-hWj9qmFT7jsYb4N791Ty9tvNx/view). Put this file in the 'yelp-weak-supervision' folder.
The pre-trained word embeddings can be downloaded at [[Download]](https://drive.google.com/file/d/1Uh7TOEqthjbzIUHIOQ2EYH1nLzVhpLrn/view). Put this file in the 'word-embedding' folder.

## Dataset

You can find the dataset in the semeval 2014 website [here](http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools).(If you cannot download the xml file, you can **`select`** the xml file, **`right-click`**, and **`save the link as`**.) Copy the dataset in the directory 'dataset'.

## Dependencies

* python 3.6.0 or higher   
* numpy 1.15.4 or higher
* gensim 3.8.3
* tqdm
* pandas
* matplotlib
* scikit-learn
* tqdm
* plotly
* nltk


## Setup: Install packages
```sh
pip install -r requirements.txt
```
### or
```sh
pip install numpy  pandas tqdm matplotlib scikit-learn tqdm plotly nltk gensim===3.8.3
```
## Run:
```sh
> python main.py
```
### then
```sh
> python demo.py
```
## File introduction
- [dataLoader.py](dataLoader.py)

This file code is used for **`dataLoad`** for loading a dataset of restaurant reviews
and related aspect-based sentiment annotations. The class uses the ElementTree library
to parse the XML data and extract sentences, labels and aspects. The extracted data
is preprocessed and stored in various class attributes such as **`train_data`**, 
**`original_train_sentences`**, **`processed_train_sentences`**, etc. The preprocessing
steps include removing digits and lemmatizing words. Some helper functions for 
vector operations such as pointwise addition, comparison, and scalar-vector 
multiplication are also defined.
 
- [PreProcessing.py](PreProcessing.py)

This file code is used for **`PreProcess`** which takes two arguments: **`Data`** and **`language`**. 
It provides text pre-processing functionality such as removing punctuation and stopwords. 
The methods used are **`Remove_Punctuation`** and **`Remove_StopWords`**.

- [xml_parser.py](xml_parser.py)

This code parses the XML file **`ABSA-15_Laptops_Train_Data.xml`** and extracts sentences and opinions. 
It then finds the unique categories of opinions and maps them to the corresponding unique category 
using the file **`laptop_map.json`**. It then splits the sentences by words and performs a word count
operation on the words in each category of sentences. The results of the word count operation are
saved to separate CSV files, one for each category.

- [model.py](model.py)

This file implements an unsupervised learning method. The model utilizes the word2vec algorithm 
to compute the similarity between words in a sentence. The data is preprocessed and loaded using 
the **`LoadDataset``** class. The class also initializes the word2vec model, the categories,
and the corresponding seed words. The **`sentence_embedd_average``** method calculates the 
average word embedding of a sentence, and the softmax method implements the softmax activation 
function to classify the sentiment of a sentence into one of the categories. The method
**`getYelpSentences`** reads and splits the raw text data into sentences, and **`similarity_matrix`** 
calculates the similarity matrix of words in the test data using the word2vec model. The similarity 
matrix is saved as a numpy file.

- [main.py](main.py)

This code performs unsupervised learning using the k-means clustering algorithm and creates 
an instance of the **`Unsupervised`** class from the **`master.model`** module and uses
it to perform k-means clustering on the target dataset. It then classifies the 
clusters and tests the performance of the classifier on a test dataset.
The **`F1-measure`**, **`precision`**, **`recall`**, and **`threshold`** values are displayed,
and the trained unsupervised model is saved for future use.

- [demo.py](demo.py)

This code is for visualizing the scatter plot of PCA transformed word embeddings.
The scatter plot is generated using Plotly's 3D scatter plot.
The scatter plot represents word embeddings in 3D space colored based on their category/label
information. The code first loads the pre-trained Unsupervised model from a pickle file
and creates PCA transformed embeddings for the seed words and the test sentences. 
The PCA transformed embeddings along with the category/label information and 
sentence information is then passed to Plotly to generate the scatter plot. 
The resulting scatter plot shows the distribution of the word embeddings in 3D space,
with the seed words and test sentences represented as points and colored based on their category/label.
The code also generates a scatter plot of the K-means clustering centroids and represents 
them in the same scatter plot with different color.



# About this project workflow:

## XML parsing and manipulation
Change dataLoader.py line 42 to your dataset.


## PCA for cluster analysis and similarity measures
The model "Unsupervised" from model.py has a method "k_means_clustering_yelp()"


### 3D plotting and visualization
After run main.py, ou can use "plotly" to plot all the centroids.



## Result
![Result](/Users/jeffereyreng/Desktop/SI699_Demo-main/屏幕录制2023-02-07-上午3.40.19.gif)

## Cite
```
@article{ghadery2018unsupervised,
  title={An Unsupervised Approach for Aspect Category Detection Using Soft Cosine Similarity Measure},
  author={Ghadery, Erfan and Movahedi, Sajad and Faili, Heshaam and Shakery, Azadeh},
  journal={arXiv preprint arXiv:1812.03361},
  year={2018}
}
```
