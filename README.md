# NLP-abstract-sentence-classification
## Problem Statement:- 
RCT papers are being published at an increasing rate; those lacking organised abstracts might be challenging to comprehend, which hinders researchers' ability to review the literature.
## Solution:- 
Creating an NLP model that categorises abstract sentences according to their functions (such as goal, methodology, findings, etc.) so that researchers can quickly scan the literature and delve further when needed.
The model will take an abstract wall of text and predict the section label each sentence should have.
## Objectives:-
1.  Downloading text dataset (PubMed RCT20k from Github).
2. Preparing the data for modelling.
3.  Setting up series for modelling experiments :- 
  Trying out different deep models with different combinations of  token embeddings,            character embeddings, pretrained embeddings, positional embeddings.
4. Comparing the accuracy of all the models (based on validation set) and finding out which works best.
5. Evaluating the best model on the test dataset.
## Notebook Contains:-
1. Preparation and Visualisation of the dataset.
2. Baseline Model
3. Model 1:- Convolutional 1D with token embeddings 
4. Model 2:- Feature extraction with pre trained token embeddings (Feature Extraction Transfer learning model)
5. Model 2:- Convolutional 1D with character embeddings
6. Model 4:- Combining pretrained token embeddings + character embeddings (hybrid embedding layer)
7. Model 5:- Tribid embedding model i.e. Transfer Learning with pretrained token embeddings + character embeddings + positional embeddings
8. Evaluating the best model on test data.

## Link to the dataset:- 
https://arxiv.org/abs/1710.06071
