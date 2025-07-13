# Clustering Fake News with K-Means and Agglomerative Clustering

This repository contains my implementation of fake news clustering, closely following the methodology from the research paper "Clustering Fake News with K-Means and Agglomerative Clustering Based on Word2Vec" (IJMCR, 2024). My code focuses on the essential steps from the paper, including cleaning news text, generating embeddings with Word2Vec, and clustering with both K-Means and Agglomerative Clustering.

## Overview

I set out to group fake and true news articles using unsupervised clustering techniques, leveraging semantic features extracted via Word2Vec embeddings. My workflow is based on:

* Text cleaning and preprocessing   
*  Word2Vec embedding generation  
*  Clustering using K-Means and Agglomerative Clustering  
* Evaluation with Purity Score and Adjusted Rand Score  

## 1\. Data Preparation & Cleaning

I began with a dataset of news articles labeled as fake or true. The cleaning steps I used are directly inspired by the research paper:

* **Remove HTML tags:** Used BeautifulSoup to strip out tags.  
*  **Normalize whitespace:** Replaced multiple spaces with a single space.  
*  **Remove numbers:** All numeric characters were removed.   
*  **Remove emojis:** Used the `emoji` library to strip out emojis.   
*  **Remove stopwords:** Used NLTK's stopword list to filter out common words.  

These steps ensure the text fed into the embedding model is as clean and meaningful as possible, just as described in the paper.

## 2\. Tokenization & Word2Vec Embeddings

After cleaning, I tokenized each article using NLTK's word and sentence tokenizers. I then trained two Word2Vec models:

*   **CBOW (Continuous Bag of Words)**    
*  **Skip-gram**  

For each article, I computed the average Word2Vec embedding by averaging the vectors of all valid tokens. This gives a dense, semantic representation of each article, following the research approach.

## 3\. Clustering

## K-Means

I applied K-Means clustering to the article embeddings, specifying 2 clusters (fake and true). Each article is assigned to the nearest centroid in the embedding space.

## Agglomerative Clustering

I also implemented Agglomerative Clustering with Ward linkage, again specifying 2 clusters. This hierarchical approach merges the most similar clusters at each step.

Both clustering methods are evaluated in the paper and in my code.

## 4\. Evaluation Metrics

To assess clustering quality, I computed two metrics as outlined in the research paper:

*   **Purity Score:** Measures the extent to which clusters contain a single class.   
*  **Adjusted Rand Score:** Measures the similarity between the clustering assignments and the true labels, adjusted for chance.   

In my experiments, K-Means generally achieved higher Purity and Adjusted Rand Scores than Agglomerative Clustering, consistent with the findings in the paper (e.g., Purity ~91% for K-Means, ~89% for Agglomerative; Adjusted Rand Score ~67% for K-Means, ~59% for Agglomerative).

## 5\. Visualization

As in the paper, I used PCA to reduce the embedding dimensions for visualization and plotted the clusters to visually inspect the separation between fake and true news articles.

## How to Run

1.  **Install dependencies:**    
    *   pandas, numpy, nltk, gensim, scikit-learn, matplotlib, emoji, beautifulsoup4
  
2.  **Prepare your dataset:**
     
    *   Place your cleaned news dataset in CSV format.
     
3.  **Run the scripts:**
    
    *   Follow the code structure in this repository to preprocess, embed, cluster, and evaluate.     *         

## Reference

*  "Clustering Fake News with K-Means and Agglomerative Clustering Based on Word2Vec," Izhar Muhammad Tianda et al., IJMCR, 2024
