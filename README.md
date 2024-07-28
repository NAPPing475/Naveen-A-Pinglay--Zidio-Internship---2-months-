# Naveen-A-Pinglay--Zidio-Internship---2-months-
Contains all the files from work done during Zidio Internship - Project 1 E- Commerce Customer Segmentation and Project 2 Emotion Recognition uisng LSTM

Project 1 - *****E Commerce Customer Segmentation using K-means Clustering*****
This project performs customer segmentation using K-Means clustering on e-commerce data, analysing customer behaviour through visualisations, and identifying key patterns for targeted marketing strategies.

This project aims to perform customer segmentation using K-Means clustering on an e-commerce dataset. The goal is to analyze customer behavior and identify key patterns for targeted marketing strategies. Below are the key steps and features of the project:

***Data Loading and Preprocessing:***

Import necessary libraries: numpy, pandas, matplotlib, seaborn, sklearn. Load e-commerce customer data from an Excel file. Clean the data by handling missing values and duplicates. Exploratory Data Analysis (EDA):

Display basic information and statistical description of the dataset. Visualize the distribution of categorical variables like gender using count plots. Analyze the distribution of numerical variables using box plots. Generate a heatmap to visualize correlations among features. Feature Engineering:

Create a new feature Total Search which sums up search counts across different brands for each customer. Visualize the top 10 customers based on total searches. Data Normalization:

Normalize the features using MinMaxScaler to prepare for clustering. K-Means Clustering:

Determine the optimal number of clusters using the elbow method and silhouette analysis. Perform K-Means clustering with the optimal number of clusters. Assign cluster labels to customers and save the clustered data to a CSV file. Cluster Analysis:

Analyze the distribution of customers across clusters using count plots. Visualize the total searches and orders for each cluster. Perform a detailed analysis of clusters 0 and 2, focusing on gender distribution and total searches. Overall Analysis:

Summarize the number of customers and their total searches and past orders in each cluster. Provide visual insights using bar plots for total searches and orders across different clusters. PDF Report:

Encode a PDF report in base64 and provide a download link for the report. By completing this project, we gain insights into customer behavior, which can be leveraged for personalized marketing strategies, enhancing customer experience, and improving business outcomes




Project 2 - *****Emotion Recognition using LSTM on Toronto Emotional Speech Set (TESS) Dataset*****

Project Overview
This project aims to recognize emotions from speech using a Long Short-Term Memory (LSTM) neural network. We use the Toronto Emotional Speech Set (TESS) dataset, which consists of recordings of actors saying the same set of phrases in different emotional states. The goal is to build and evaluate a model that can accurately classify the emotion conveyed in the speech.

Key Steps and Features of the Project:



***1. Data Loading and Preprocessing***

Import Necessary Libraries: Import essential libraries such as pandas, numpy, librosa, seaborn, matplotlib, and Keras.
Download and Unzip Dataset: Use the Kaggle API to download and unzip the TESS dataset.
Data Collection: Traverse the dataset directory to collect file paths and corresponding labels (emotions).
Create DataFrame: Create a DataFrame to store the file paths and labels.

***2. Exploratory Data Analysis (EDA)***
Label Distribution Visualization: Use count plots to visualize the distribution of different emotion labels in the dataset.
Waveplot and Spectrogram: Define functions to plot the waveform and spectrogram of audio files to understand the differences in audio characteristics for different emotions.

***3. Feature Extraction***
Label Encoding: Encode the categorical emotion labels into numerical format and convert them to categorical format for model training.

***4. Model Training with LSTM***
Data Splitting: Split the data into training and testing sets.
Data Reshaping: Reshape the data to fit the input requirements of the LSTM model.
Build LSTM Model: Define an LSTM model with masking, LSTM, dropout, and dense layers.
Model Compilation: Compile the model using categorical cross-entropy loss and Adam optimizer.
Model Training: Train the model on the training data and validate it on the validation set.

***5. Model Evaluation***
Model Evaluation: Evaluate the model on the test set and print the accuracy.
Training History Visualization: Plot the training and validation accuracy and loss over epochs to analyze the model's performance.

***6.Visualization of Results***
The code evaluates the model's performance on test data and prints the accuracy. It then plots training and validation accuracy as well as loss over epochs. This visual representation helps in understanding the model's performance and how it evolves during training.
