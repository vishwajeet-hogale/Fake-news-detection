# Hindi Fake News Detection Notebook Documentation
## Overview
This Jupyter notebook is designed for the task of detecting fake news in Hindi. It covers the entire pipeline from data loading and preprocessing to training machine learning models and evaluating their performance.

## Sections
### Data Loading and Exploration
* Load the true and fake news datasets from CSV files.
* Display basic information about the datasets.

### Data Cleaning
* Check for missing values in the datasets.
* Perform basic data exploration.

### Data Preprocessing
* Remove punctuations, emojis, dates, links, emails, URLs, and numbers from the articles.
* Tokenize and lemmatize the articles.
* Remove stopwords.
* Save the cleaned data to new CSV files.

### Vectorization
* Use TF-IDF Vectorizer for feature extraction.

### Model Training
#### Train multiple models:
* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (RBF Kernel)
* Support Vector Machine (Linear Kernel)
* Passive Aggressive Classifier
* Artificial Neural Network (ANN)

### Model Evaluation
* Evaluate the models using metrics like accuracy, precision, recall, F1 score, Cohen's kappa score, ROC AUC, and confusion matrix.
* Save the results to a CSV file.

### Word Embeddings and Visualization
* Use Word2Vec for word embeddings.
* Visualize the embeddings using t-SNE.

### Translation and Prediction
* Use Google Translate API for translating English text to Hindi.
* Utilize the trained models for predicting whether the translated text contains fake news or not.

### Additional Neural Network Models
* Train additional neural network models using LSTM and CNN architectures.
* Plot training and validation loss and accuracy.
### Word Cloud
* Generate a word cloud to visualize the most frequent words in the preprocessed articles.

## Models
* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (RBF Kernel)
* Support Vector Machine (Linear Kernel)
* Passive Aggressive Classifier
* Artificial Neural Network (ANN)
* LSTM (Long Short-Term Memory) Neural Network
* ANN (Artificial Neural Network)

## Data Files
* Cleaned_hindi_news.csv: CSV file containing cleaned true news articles.
* Cleaned_hindi_fake.csv: CSV file containing cleaned fake news articles.
* NewTrueh.csv: CSV file containing true news articles and corresponding labels.
* NewFakeh.csv: CSV file containing fake news articles and corresponding labels.
* cleaned.csv: CSV file containing cleaned and preprocessed articles.
* cleaned_hindi.csv: CSV file containing cleaned Hindi articles after preprocessing.
* finalresults2.csv: CSV file containing the evaluation results of various models.
* loss_plot.png: Plot showing the training and validation loss of the LSTM model.
* accuracy_plot.png: Plot showing the training and validation accuracy of the LSTM model.
* bow1.png: t-SNE plot of Word2Vec embeddings.

## Libraries Used
* pandas
* numpy
* matplotlib
* scikit-learn
* nltk
* tensorflow
* gensim
* googletrans
* wordcloud

## Instructions for Use
* Upload the datasets (Cleaned_hindi_news.csv and Cleaned_hindi_fake.csv) to the Colab environment.
* Run the cells sequentially to execute the entire pipeline.
* View the generated visualizations and evaluation results.
## Note
Ensure that the necessary libraries are installed in the Colab environment before running the notebook.







