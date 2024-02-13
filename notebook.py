#!/usr/bin/env python
# coding: utf-8

import os
import re
import tqdm
from glob import glob
from pathlib import Path
import random as rnd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pprint import pprint
import graphviz

from collections import defaultdict
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import xgboost as xgb

import time
from textblob import TextBlob


# # Preprocessing

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.strip().lower()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # Removing non-alphanumeric characters
    text = re.sub(r"\b\d+\b", "<NUM>", text) # Replacing numeric values with <NUM>

    words = text.split()

    words = [word for word in words if word not in stop_words]  # Removing stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization

    return " ".join(words)


data_path = "Data/dataset/*.html"

code2convos = dict()

pbar = tqdm.tqdm(sorted(list(glob(data_path))))
for path in pbar:
    file_code = os.path.basename(path).split(".")[0]
    with open(path, "r", encoding="latin1") as fh:
            
        # Getting the file ID to use it as a key later on
        fid = os.path.basename(path).split(".")[0]

        # Reading the HTML file
        html_page = fh.read()

        # Parsing the HTML file with bs4
        soup = BeautifulSoup(html_page, "html.parser")

        # Grabbing the conversations with the data-testid pattern
        data_test_id_pattern = re.compile(r"conversation-turn-[0-9]+")
        conversations = soup.find_all("div", attrs={"data-testid": data_test_id_pattern})

        convo_texts = []

        for i, convo in enumerate(conversations):
            convo = convo.find_all("div", attrs={"data-message-author-role":re.compile( r"[user|assistant]") })
            if len(convo) > 0:
                role = convo[0].get("data-message-author-role")

                # PREPROCESSING
                text = convo[0].text
                processed_text = preprocess_text(text)

                convo_texts.append({
                        "role" : role,
                        "text" : processed_text
                    }
                )
                
        code2convos[file_code] = convo_texts


# Sample entry
pprint(code2convos)


# # Matching Prompts with Questions
# We matched the prompts with the questions in the homework using the code cells below. Firstly, for each student, the prompts were vectorized with a TF-IDF vectorizer, with the empty prompts marked and excluded along the way. Then, the same was done for the questions. A cosine similarity matrix was constructed using the two.
# 
# The similarity matrix was used for outlier treatment. Codes (identifying students) associated with less than 15% similarity or contained empty prompts were eliminated.
# 
# Finally, the treated data was vectorized with Word2Vec in 2 different ways: 1) per student and 2) as a whole, once. The resulting data was used to build another cosine similarity matrix. The final similarity matrix as well as the two Word2Vec vectors formed the basis the dataset we used for the machine learning algorithms.

# Creating a list of user prompts
prompts = []
code2prompts = defaultdict(list)
for code, convos in code2convos.items():
    user_prompts = []
    for conv in convos:
        if conv["role"] == "user":
            prompts.append(conv["text"])
            user_prompts.append(conv["text"])
    code2prompts[code] = user_prompts

pprint(code2prompts["0031c86e-81f4-4eef-9e0e-28037abf9883"])


questions = [
    """Initialize
*   First make a copy of the notebook given to you as a starter.
*   Make sure you choose Connect form upper right.
*   You may upload the data to the section on your left on Colab, than right click on the .csv file and get the path of the file by clicking on "Copy Path". You will be using it when loading the data.

""",
#####################
    """Load training dataset (5 pts)
    *  Read the .csv file with the pandas library
""",
#####################
"""Understanding the dataset & Preprocessing (15 pts)
Understanding the Dataset: (5 pts)
> - Find the shape of the dataset (number of samples & number of attributes). (Hint: You can use the **shape** function)
> - Display variable names (both dependent and independent).
> - Display the summary of the dataset. (Hint: You can use the **info** function)
> - Display the first 5 rows from training dataset. (Hint: You can use the **head** function)
Preprocessing: (10 pts)

> - Check if there are any missing values in the dataset. If there are, you can either drop these values or fill it with most common values in corresponding rows. **Be careful that you have enough data for training the  model.**

> - Encode categorical labels with the mappings given in the cell below. (Hint: You can use **map** function)
""",
"""Set X & y, split data (5 pts)

*   Shuffle the dataset.
*   Seperate your dependent variable X, and your independent variable y. The column health_metrics is y, the rest is X.
*   Split training and test sets as 80% and 20%, respectively.
""",
#####################
"""Features and Correlations (10 pts)

* Correlations of features with health (4 points)
Calculate the correlations for all features in dataset. Highlight any strong correlations with the target variable. Plot your results in a heatmap.

* Feature Selection (3 points)
Select a subset of features that are likely strong predictors, justifying your choices based on the computed correlations.

* Hypothetical Driver Features (3 points)
Propose two hypothetical features that could enhance the model's predictive accuracy for Y, explaining how they might be derived and their expected impact. Show the resulting correlations with target variable.

* __Note:__ You get can get help from GPT.
""",
#####################
"""Tune Hyperparameters (20 pts)
* Choose 2 hyperparameters to tune. You can use the Scikit learn decision tree documentation for the available hyperparameters *(Hyperparameters are listed under "Parameters" in the documentation)*. Use GridSearchCV for hyperparameter tuning, with a cross-validation value of 5. Use validation accuracy to pick the best hyper-parameter values. (15 pts)
-Explain the hyperparameters you chose to tune. *(What are the hyperparameters you chose? Why did you choose them?)* (5 pts)
""",
#####################
"""Re-train and plot the decision tree with the hyperparameters you have chosen (15 pts)
- Re-train model with the hyperparameters you have chosen in part 5). (10 pts)
- Plot the tree you have trained. (5 pts)
Hint: You can import the **plot_tree** function from the sklearn library.
""",
#####################
"""Test your classifier on the test set (20 pts)
- Predict the labels of testing data using the tree you have trained in step 6. (10 pts)
- Report the classification accuracy. (2 pts)
- Plot & investigate the confusion matrix. Fill the following blanks. (8 pts)
> The model most frequently mistakes class(es) _________ for class(es) _________.
Hint: You can use the confusion_matrix function from sklearn.metrics
""",
#####################
"""Find the information gain on the first split (10 pts)""",
#####################
]


# Preprocessing the questions
questions = [preprocess_text(question) for question in questions]


# ## Vectorising with TF-IDF

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(prompts + questions)


questions_TF_IDF = pd.DataFrame(vectorizer.transform(questions).toarray(), columns=vectorizer.get_feature_names_out())
questions_TF_IDF.head()


# Creating a TF-IDF model for each student and a dictionary that maps each student code to its
# TF-IDF matrix (formed with the corresponding prompts)
code2prompts_TF_IDF = dict()
codes_with_no_prompts = []

for code, user_prompts in code2prompts.items():
    if len(user_prompts) == 0: # Excluding empty prompts and adding the codes associated with them to a list (to eliminate them from scores later)
        print(code+".html")
        codes_with_no_prompts.append(code)
        continue
    prompts_TF_IDF = pd.DataFrame(vectorizer.transform(user_prompts).toarray(), columns=vectorizer.get_feature_names_out())
    code2prompts_TF_IDF[code] = prompts_TF_IDF

print(code2prompts_TF_IDF["089eb66d-4c3a-4f58-b98f-a3774a2efb34"].shape)
code2prompts_TF_IDF["089eb66d-4c3a-4f58-b98f-a3774a2efb34"].head()


# Demonstrating the structure of the dictionary
print("The user with code " + list(code2prompts_TF_IDF.keys())[0] + " has " + str(len(code2prompts_TF_IDF[list(code2prompts_TF_IDF.keys())[0]])) + " prompts.")
print("The first three prompts have the following TF-IDF matrices:\n" + str(code2prompts_TF_IDF[list(code2prompts_TF_IDF.keys())[0]].head(3)))


# Creating a dictionary that pairs each student code with the corresponding 
# cosine similarity (between prompts and questions) vector
code2cosine = dict()
for code, user_prompts_TF_IDF in code2prompts_TF_IDF.items():
    code2cosine[code] = pd.DataFrame(cosine_similarity(questions_TF_IDF,user_prompts_TF_IDF))

print(code2cosine["089eb66d-4c3a-4f58-b98f-a3774a2efb34"].shape)
code2cosine["089eb66d-4c3a-4f58-b98f-a3774a2efb34"].head()


# Taking the highest cosine similarity value from each row (i.e. for each question) for each student and assigning the resulting vector to the student
code2questionmapping = dict()
for code, cosine_scores in code2cosine.items():
    code2questionmapping[code] = code2cosine[code].max(axis=1).tolist()

# Sample cosine similarity vector for a student
print(code2questionmapping["089eb66d-4c3a-4f58-b98f-a3774a2efb34"])

# Creating a dataframe that displays the mapping better
question_mapping_scores = pd.DataFrame(code2questionmapping).T
question_mapping_scores.reset_index(inplace=True)
question_mapping_scores.rename(columns={i: f"Q_{i}" for i in range(len(questions))}, inplace=True)
question_mapping_scores.rename(columns={"index": "code"}, inplace=True)

# Displaying the dataframe
print(question_mapping_scores)


# # Outlier Treatment

# ## Dropping Outliers and Students with Empty Prompts (Preferred Approach)

# Outliers [in this context]: Dummy or highly inconsistent prompts

# Pipeline for detecting an outlier (an example):
# Filtering the dataframe based on the desired code
desired_code = "6a2003ad-a05a-41c9-9d48-e98491a90499"
desired_file_scores = question_mapping_scores[question_mapping_scores['code'] == desired_code]

# Calculating the average distance for the specific code
average_distance = desired_file_scores.iloc[:, 1:].mean(axis=1).values[0]

# Printing the average distance
print(f"The average distance for code {desired_code} is: {average_distance}")

threshold = 0.15

question_mapping_scores['average_distance'] = question_mapping_scores.iloc[:, 1:].mean(axis=1)

filtered_scores = question_mapping_scores[question_mapping_scores['average_distance'] >= threshold].reset_index(drop=True)

filtered_scores = filtered_scores.drop(columns=['average_distance']).reset_index(drop=True)

print(f"Number of rows in filtered_scores: {len(filtered_scores)}")

dropped_codes = question_mapping_scores[~question_mapping_scores['code'].isin(filtered_scores['code'])]['code'].tolist()

# Print the codes of dropped entries
print("Codes of dropped entries:")
print(dropped_codes)

question_mapping_scores = filtered_scores


# ## Data Imputation with kNN (Alternative Approach)

# Outliers [in this context]: Dummy or highly inconsistent prompts

# Calculating the average distance for each code
question_mapping_scores['average_distance'] = question_mapping_scores.iloc[:, 1:].mean(axis=1)

# Setting the threshold for outliers
threshold = 0.15

# Identifying outliers
outliers = question_mapping_scores[question_mapping_scores['average_distance'] < threshold]

# Setting up the k-NN model
k = 5  # Number of neighbors; you can adjust this number as needed
nn_model = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included

# Training the model with non-outlier data
non_outliers = question_mapping_scores[question_mapping_scores['average_distance'] >= threshold]
nn_model.fit(non_outliers.iloc[:, 1:-1])  # Exclude 'code' and 'average_distance' columns

# For each outlier, find its nearest neighbors and replace its data
for index, outlier in outliers.iterrows():
    distances, indices = nn_model.kneighbors([outlier.iloc[1:-1]], n_neighbors=k+1)

    # Exclude the first neighbor as it is the outlier itself
    nearest_neighbors = non_outliers.iloc[indices[0][1:], 1:-1]

    # Replace outlier data with the mean of its nearest neighbors
    question_mapping_scores.loc[index, 1:-1] = nearest_neighbors.mean(axis=0)

# Dropping the average_distance column as it's no longer needed
question_mapping_scores = question_mapping_scores.drop(columns=['average_distance'])

# Print the number of rows after processing
print(f"Number of rows after processing: {len(question_mapping_scores)}")


# ## Vectorising with Word2Vec

# TF-IDF looks for direct word matches; word2vec tries to infer the context
nltk.download('punkt')

# Creating a word2vec model using all prompts
tokenized_prompts = [word_tokenize(prompt) for prompt in prompts if len(prompt) > 0]
tokenized_questions = [word_tokenize(question) for question in questions if len(question) > 0]

model = Word2Vec(tokenized_prompts + tokenized_questions, vector_size=500, window=5, min_count=1, workers=4)

prompts_embeddings = [np.mean([model.wv[word] for word in prompt], axis=0) for prompt in tokenized_prompts]
questions_embeddings = [np.mean([model.wv[word] for word in question], axis=0) for question in tokenized_questions]

code2cosine_word2vec = dict()
for code, prompt_embedding in zip(code2prompts.keys(), prompts_embeddings):
    code2cosine_word2vec[code] = [cosine_similarity([question_embedding], [prompt_embedding])[0][0] for question_embedding in questions_embeddings]

code2questionmapping_word2vec = pd.DataFrame(code2cosine_word2vec).T
code2questionmapping_word2vec.reset_index(inplace=True, drop=False)
code2questionmapping_word2vec.rename(columns={i: f"Q_{i}" for i in range(len(questions))}, inplace=True)
code2questionmapping_word2vec.rename(columns={"index": "code"}, inplace=True)

zero_length_prompts_codes = [code for code, prompt in code2prompts.items() if len(prompt) == 0]
code2questionmapping_word2vec_filtered = code2questionmapping_word2vec[~code2questionmapping_word2vec['code'].isin(zero_length_prompts_codes)]

question_mapping_scores = code2questionmapping_word2vec_filtered


# Creating a word2vec model for each student
columns = ["code"] + ["w" + str(i) for i in range(500)]
code2word2vec = pd.DataFrame(columns=columns)

for code in code2prompts:
    if len(code2prompts[code]) != 0:
        model_w2v = Word2Vec(code2prompts[code], vector_size=500, window=5, min_count=1, workers=4)
        
        prompt_embeddings = []
        for prompt in code2prompts[code]:
            word_vectors = [model_w2v.wv[word] for word in prompt if word in model_w2v.wv]
            if word_vectors:  # Checking if the list is not empty
                average_vector = np.mean(word_vectors, axis=0)
                prompt_embeddings.append(average_vector)
            else:
                pass
        
        if prompt_embeddings:
            code_embedding = np.mean(prompt_embeddings, axis=0)

            row = [code] + code_embedding.tolist()
            code2word2vec = code2word2vec.append(pd.Series(row, index=columns), ignore_index=True)
    else:
        pass

print(code2word2vec.shape)
code2word2vec.head()


# ## Normalization

# Data was normalized using Min-Max scaling. The normalized data was used only for neural network training; non-normalized data was used for other algorithms.

# Reading the scores
scores = pd.read_csv("Data/scores.csv", sep=",")
scores["code"] = scores["code"].apply(lambda x: x.strip())

scores = scores[["code", "grade"]]

# Removing grades received by outlier students
scores = scores[~scores["code"].isin(dropped_codes)].reset_index(drop=True)
scores = scores[~scores["code"].isin(codes_with_no_prompts)].reset_index(drop=True)
scores = scores[scores["grade"] >= 70].reset_index(drop=True)

# Displaying some scores
print(scores.shape)
scores.head()


# The distribution of grades
plt.title('Histogram Grades')
plt.hist(scores["grade"], rwidth=.8, bins=np.arange(min(scores["grade"]), max(scores["grade"])+2) - 0.5)
plt.ylabel('Count')
plt.show()


# Min-Max scaling
scaler = MinMaxScaler()

# Scaling the scores
scaled_values = scaler.fit_transform(scores[scores.columns[1:]])

# Converting the scaled values back into a dataframe
normalized_scores = pd.DataFrame(scaled_values, columns=scores.columns[1:])

# Merging the 'code' column from the original 'scores' dataframe with the new 'normalized_scores'
normalized_scores = pd.merge(scores[["code"]], normalized_scores, left_index=True, right_index=True)

print(normalized_scores)


# # Feature Engineering
# Possibly predictive features:
# * Number of prompts that a user asked
# * Number of complaints that a user made e.g "the code gives this error!"
# * The average number of characters in a user's prompts
# * Whether context was given in the first prompt
# * Number of apologies by ChatGPT
# * Number of ideal student keywords (words used by the best students)
# * Sentiment score

# Finding the best students (those who scored 100)
best_scores = scores[scores["grade"] == 100]

best_students = best_scores["code"].tolist()
print(len(best_students))
print("The best students are: " + str(best_students))


prompts_of_best_students = [code2prompts[code] for code in best_students]

# Finding words most frequently used by the best students
counts = {}
for prompts in prompts_of_best_students:
    for prompt in prompts:
        prompt_list = prompt.split()
        for word in prompt_list:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1

# Sorting the words by their counts
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
print("Top 10 words in best students' prompts: " + str(sorted_counts[1:11]))

sorted_keys = sorted(counts, key=counts.get, reverse=True)
best_words = sorted_keys[1:11] #first one is num, so skip that
print("Top 10 words in best students' prompts: " + str(best_words))


def is_context_in_first_prompt(prompts):
    if len(prompts) == 0:
        return 0
    
    first_prompt= prompts[0]

    context_words = ["python", "machine learning", "computer science", "cs", "decision tree", "scikit-learn", "classifier", "classification", "classifiers", "classifications", "trees", "tree", "decision"]

    for word in context_words:
        if word in first_prompt:
            return 1
    return 0


code2features = defaultdict(lambda : defaultdict(int))

keywords2search = ["error", "no", "thank", "Entropy"]
keywords2search = [k.lower() for k in keywords2search]

apology_keywords = ["sorry", "apologize", "apology", "apologies", "oversight"]
apology_keywords = [k.lower() for k in apology_keywords]

for code, convs in code2convos.items():
    if len(convs) == 0:
        print(code)
        continue
    for c in convs:
        text = c["text"].lower()
        if c["role"] == "user": # User
            # Counting user prompts
            code2features[code]["#user_prompts"] += 1

            sentiment_score = TextBlob(text).sentiment.polarity
            code2features[code]["sentiment_score"] += sentiment_score

            # Counting the keywords
            for kw in keywords2search:
                code2features[code][f"#{kw}"] +=  len(re.findall(rf"\b{kw}\b", text))
            
            # Counting ideal words
            for bw in best_words:
                code2features[code][f"#{bw}"] +=  len(re.findall(rf"\b{bw}\b", text))

            # Determining whether context was provided in the first prompt
            code2features[code]["context_in_first_prompt"] = is_context_in_first_prompt(code2prompts[code])

            code2features[code]["prompt_avg_chars"] += len(text)
        else: # ChatGPT
            code2features[code]["response_avg_chars"] += len(text)

            # Counting apologies
            for kw in apology_keywords:
                code2features[code][f"#{kw}"] +=  len(re.findall(rf"\b{kw}\b", text))

        code2features[code]["prompt_avg_chars"] /= code2features[code]["#user_prompts"]
        code2features[code]["response_avg_chars"] /= code2features[code]["#user_prompts"]


# Gathering the features in a dataframe
df = pd.DataFrame(code2features).T
df.head(5)


df.reset_index(inplace=True, drop=False)
df.rename(columns={"index": "code"}, inplace=True)
df.head()


# Integrating mapping scores to the dataframe
df = pd.merge(df, question_mapping_scores, on="code", how="left")

print(df.shape)
df.head()


# #### Merging Scores with Features
# 4 datasets were created:
# * Engineereed features + highest cosine similarity for each question + regular scores
# * Engineered features + highest cosine similarity for each question + word2vec vector + regular scores
# * Engineered features + highest cosine similarity for each question + normalized scores
# * Engineered features + highest cosine similarity for each question + word2vec vector + normalized scores

# Merging the newly created dataframe with regular scores
temp_df_regular = pd.merge(df, scores, on="code", how="left")
temp_df_regular.dropna(inplace=True)
temp_df_regular.drop_duplicates("code", inplace=True, keep="first")
temp_df_regular.head()

temp_df_sorted_regular = temp_df_regular.sort_values(by="grade", ascending=False)

# Printing the sorted dataframe
print(temp_df_sorted_regular)


# Merging the newly created dataframe with separate word2vec models and regular scores
temp_df_regular_extra_1 = pd.merge(df, code2word2vec, on="code", how="left")
temp_df_regular_extra_1 = pd.merge(temp_df_regular_extra_1, scores, on="code", how="left")
temp_df_regular_extra_1.dropna(inplace=True)
temp_df_regular_extra_1.drop_duplicates("code", inplace=True, keep="first")
temp_df_regular_extra_1.head()

temp_df_sorted_regular_extra_1 = temp_df_regular_extra_1.sort_values(by="grade", ascending=False)

# Printing the sorted dataframe
print(temp_df_sorted_regular_extra_1)


# Merging the newly created dataframe with normalized scores
temp_df_normalized = pd.merge(df, normalized_scores, on="code", how="left")
temp_df_normalized.dropna(inplace=True)
temp_df_normalized.drop_duplicates("code",inplace=True, keep="first")
temp_df_normalized.head()

temp_df_sorted_normalized = temp_df_normalized.sort_values(by="grade", ascending=False)

# Printing the sorted dataframe
print(temp_df_sorted_normalized)


# Merging the newly created dataframe separate word2vec models and normalized scores
temp_df_normalized_extra_1 = pd.merge(df, code2word2vec, on="code", how="left")
temp_df_normalized_extra_1 = pd.merge(temp_df_normalized_extra_1, normalized_scores, on="code", how="left")
temp_df_normalized_extra_1.dropna(inplace=True)
temp_df_normalized_extra_1.drop_duplicates("code",inplace=True, keep="first")
temp_df_normalized_extra_1.head()

temp_df_sorted_normalized_extra_1 = temp_df_normalized_extra_1.sort_values(by="grade", ascending=False)

# Printing the sorted dataframe
print(temp_df_sorted_normalized_extra_1)


X = temp_df_regular[temp_df_regular.columns[1:-1]].to_numpy()
y = temp_df_regular["grade"].to_numpy()
print(X.shape, y.shape)


X_extra_1 = temp_df_regular_extra_1[temp_df_regular_extra_1.columns[1:-1]].to_numpy()
y_extra_1 = temp_df_regular_extra_1["grade"].to_numpy()
print(X_extra_1.shape, y_extra_1.shape)


X_normalized = temp_df_normalized[temp_df_normalized.columns[1:-1]].to_numpy()
y_normalized = temp_df_normalized["grade"].to_numpy()
print(X_normalized.shape, y_normalized.shape)


X_normalized_extra_1 = temp_df_normalized_extra_1[temp_df_normalized_extra_1.columns[1:-1]].to_numpy()
y_normalized_extra_1 = temp_df_normalized_extra_1["grade"].to_numpy()
print(X_normalized_extra_1.shape, y_normalized_extra_1.shape)


# #### Train/Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train set size:", len(X_train))
print("Test set size:", len(X_test))


X_train_extra_1, X_test_extra_1, y_train_extra_1, y_test_extra_1 = train_test_split(X_extra_1, y_extra_1, test_size=0.2, random_state=42)
print("Train set size:", len(X_train_extra_1))
print("Test set size:", len(X_test_extra_1))


X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
print("Train set size:", len(X_train_normalized))
print("Test set size:", len(X_test_normalized))


X_train_normalized_extra_1, X_test_normalized_extra_1, y_train_normalized_extra_1, y_test_normalized_extra_1 = train_test_split(X_normalized_extra_1, y_normalized_extra_1, test_size=0.2, random_state=42)
print("Train set size:", len(X_train_normalized_extra_1))
print("Test set size:", len(X_test_normalized_extra_1))


# # Machine Learning

# ## Neural Networks

# Building the neural network
model_NN_1 = Sequential()

# Input layer
model_NN_1.add(Dense(128, input_shape=(X_train_normalized.shape[1],), activation='relu'))

# Hidden layers
model_NN_1.add(Dense(64, activation='relu'))
model_NN_1.add(Dropout(0.7))

# Output layer
model_NN_1.add(Dense(1, activation='linear'))

# Compiling the model
adam_optimizer = Adam(learning_rate=0.01)
model_NN_1.compile(optimizer=adam_optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Displaying the model summary
model_NN_1.summary()

# Training the model with a validation split
history = model_NN_1.fit(
    X_train_normalized, y_train_normalized,
    epochs=500,
    batch_size=32,
    validation_split=0.20,
    shuffle=False
)

normalized_predictions = model_NN_1.predict(X_train_normalized)

# Inverse transforming predictions and validation labels
predictions = scaler.inverse_transform(normalized_predictions)
y_valid_original = scaler.inverse_transform(y_train_normalized.reshape(-1, 1))

# Calculating MAE and MSE on the original scale
training_mae_1 = mean_absolute_error(y_valid_original, predictions)
training_mse_1 = mean_squared_error(y_valid_original, predictions)

print(f"Mean absolute error: {training_mae_1}")
print(f"Mean squared error: {training_mse_1}")


normalized_predictions = model_NN_1.predict(X_test_normalized)

# Inverse transforming predictions and validation labels
predictions = scaler.inverse_transform(normalized_predictions)
y_valid_original = scaler.inverse_transform(y_test_normalized.reshape(-1, 1))

# Calculating MAE and MSE on the original scale
test_mae_1 = mean_absolute_error(y_valid_original, predictions)
test_mse_1 = mean_squared_error(y_valid_original, predictions)

print(f"Mean absolute error: {test_mae_1}")
print(f"Mean squared error: {test_mse_1}")


# Building the neural network
model_NN_2 = Sequential()

# Input layer
model_NN_2.add(Dense(128, input_shape=(X_train_normalized_extra_1.shape[1],), activation='relu'))

# Hidden layers
model_NN_2.add(Dense(64, activation='relu'))
model_NN_2.add(Dropout(0.7))

# Output layer
model_NN_2.add(Dense(1, activation='linear'))

# Compiling the model
adam_optimizer = Adam(learning_rate=0.01)
model_NN_2.compile(optimizer=adam_optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Displaying the model summary
model_NN_2.summary()

# Training the model with a validation split
history = model_NN_2.fit(
    X_train_normalized_extra_1, y_train_normalized_extra_1,
    epochs=500,
    batch_size=32,
    validation_split=0.20,
    shuffle=False
)

normalized_predictions = model_NN_2.predict(X_train_normalized_extra_1)

# Inverse transforming predictions and validation labels
predictions = scaler.inverse_transform(normalized_predictions)
y_valid_original = scaler.inverse_transform(y_train_normalized_extra_1.reshape(-1, 1))

# Calculating MAE and MSE on the original scale
training_mae_2 = mean_absolute_error(y_valid_original, predictions)
training_mse_2 = mean_squared_error(y_valid_original, predictions)

print(f"Mean absolute error: {training_mae_2}")
print(f"Mean squared error: {training_mse_2}")


normalized_predictions = model_NN_2.predict(X_test_normalized_extra_1)

# Inverse transforming predictions and validation labels
predictions = scaler.inverse_transform(normalized_predictions)
y_valid_original = scaler.inverse_transform(y_test_normalized_extra_1.reshape(-1, 1))

# Calculating MAE and MSE on the original scale
test_mae_2 = mean_absolute_error(y_valid_original, predictions)
test_mse_2 = mean_squared_error(y_valid_original, predictions)

print(f"Mean absolute error: {test_mae_2}")
print(f"Mean squared error: {test_mse_2}")


# ## Clustering

# Range of k values to try
k_values = range(2, len(set(scores["grade"])) + 1)  # Adjust this range based on your dataset

sum_of_squared_distances = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_extra_1)
    sum_of_squared_distances.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, sum_of_squared_distances, 'bx-')
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.title("Elbow Method For Optimal k")
plt.xticks(k_values)
plt.show()


clustered_temp_df_regular_extra_1 = temp_df_regular_extra_1

# Choosing the number of clusters
num_clusters = 6

# Applying K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(X_extra_1)

# Adding the cluster labels to the newly created replicate dataframe
clustered_temp_df_regular_extra_1['cluster'] = clusters

clustered_temp_df_regular_extra_1.head()


# ## Decision Tree

regressor = DecisionTreeRegressor(random_state=0,criterion='squared_error', max_depth=10)
regressor.fit(X_train, y_train)


################################################################
################################################################
##############       HYPERPARAMETER TUNING      ################
################################################################
################################################################

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# setup parameter space
parameters = {'criterion':['squared_error'],
              'max_depth':np.arange(1,21).tolist()[0::2],
              'min_samples_split':np.arange(2,11).tolist()[0::2],
              'max_leaf_nodes':np.arange(3,26).tolist()[0::2]}

# create an instance of the grid search object
g2 = GridSearchCV(DecisionTreeRegressor(), parameters, cv=5, n_jobs=-1)

# conduct grid search over the parameter space
start_time = time.time()
g2.fit(X_train,y_train)
duration = time.time() - start_time

# show best parameter configuration found for regressor
rgr_params1 = g2.best_params_
rgr_params1


extracted_MSEs = regressor.tree_.impurity   
for idx, MSE in enumerate(regressor.tree_.impurity):
    print("Node {} has MSE {}".format(idx,MSE))


# Plotting the Tree 
dot_data = tree.export_graphviz(regressor, out_file=None, feature_names=temp_df_regular.columns[1:-1])
graph = graphviz.Source(dot_data)
graph.render("hw")


# Prediction
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# Calculation of Mean Absolute Error (MAE), Mean Squared Error (MSE), and R2 score
training_mae_3 = mean_squared_error(y_train, y_train_pred)
test_mae_3 = mean_absolute_error(y_test, y_test_pred)
training_mse_3 = mean_absolute_error(y_train, y_train_pred)
test_mse_3 = mean_squared_error(y_test, y_test_pred) 
training_r2_score_1 = r2_score(y_train, y_train_pred)
test_r2_score_1 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {training_mse_3}")
print(f"Test MSE: {test_mse_3}")
print(f"Training MAE: {training_mae_3}")
print(f"Test MAE: {test_mae_3}")

print(f"Training R2: {training_r2_score_1}")
print(f"Test R2: {test_r2_score_1}")


# ## Random Forest

################################################################
################################################################
##########         RandomForestRegressor APPROACH      #########
################################################################
################################################################

# Extracting features (X) and target variable (y)
X = temp_df_regular[temp_df_regular.columns[1:-1]].to_numpy()
y = temp_df_regular["grade"].to_numpy()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("Train set size:", len(X_train))
print("Test set size:", len(X_test))

# Setting up the parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initializing the RandomForestRegressor
regressor = RandomForestRegressor(random_state=0)

# Creating the GridSearchCV object
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fitting the model with the grid search parameters
grid_search.fit(X_train, y_train)

# Getting the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Making predictions on the test set using the best model
best_regressor = grid_search.best_estimator_
test_predictions = best_regressor.predict(X_test)
train_predictions = best_regressor.predict(X_train)

# Evaluating the model
training_mae_4 = mean_absolute_error(y_train, train_predictions)
test_mae_4 = mean_absolute_error(y_test, test_predictions)
training_mse_4 = mean_squared_error(y_train, train_predictions)
test_mse_4 = mean_squared_error(y_test, test_predictions)

print(f"Training MAE: {training_mae_4}")
print(f"Test MAE: {test_mae_4}")
print(f"Training MSE: {training_mse_4}")
print(f"Test MSE: {test_mse_4}")


# ## XGBoost

################################################################
################################################################
#############          XGBOOST APPROACH           ##############
################################################################
################################################################

X = temp_df_regular.drop(columns=['grade'])  # Features
y = temp_df_regular['grade']  # Target variable

print(temp_df_regular.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.drop(columns=['code'])
X_test = X_test.drop(columns=['code'])


model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE Train:", mean_squared_error(y_train,model.predict(X_train)))
print("MSE TEST:", mean_squared_error(y_test,y_pred))

print("R2 Train:", r2_score(y_train,model.predict(X_train)))
print("R2 TEST:", r2_score(y_test,y_pred))


param_grid = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Printing the best hyperparameters
print("Best Hyperparameters:")
print(best_params)

# Making predictions using the best model
final_predictions = best_model.predict(X_test)

# Evaluating the best model
mae = mean_absolute_error(y_test, final_predictions)
mse = mean_squared_error(y_test, final_predictions)
r2 = r2_score(y_test, final_predictions)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


X = temp_df_regular.drop(columns=['grade'])  # Features
y = temp_df_regular['grade']  # Target variable

# Droppinf 'code' column before splitting
X = X.drop(columns=['code'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# K-fold cross-validation
cv_results = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Displaying cross-validation results
print("Cross-validation Results:")
print("Negative Mean Squared Errors:", cv_results)
print("Average Negative Mean Squared Error:", cv_results.mean())

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Getting the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Printing the best hyperparameters
print("\nBest Hyperparameters:")
print(best_params)

# Making predictions using the best model
test_final_predictions = best_model.predict(X_test)
train_final_predictions = best_model.predict(X_train)

# Evaluating the best model
training_mae_5 = mean_absolute_error(y_train, train_final_predictions)
test_mae_5 = mean_absolute_error(y_test, test_final_predictions)
training_mse_5 = mean_squared_error(y_train, train_final_predictions)
test_mse_5 = mean_squared_error(y_test, test_final_predictions)
test_r2_score_2 = r2_score(y_test, test_final_predictions)

print(f"Training MAE: {training_mae_5}")
print(f"Test MAE: {test_mae_5}")
print(f"Training MSE: {training_mse_5}")
print(f"Test MSE: {test_mse_5}")
print(f"Test R2 score: {test_r2_score_2}")


# ## Dummy Regressors

temp_df_above_70 = temp_df_regular[temp_df_regular["grade"] >= 70]

X = temp_df_above_70[temp_df_above_70.columns[1:-1]].to_numpy()
y = temp_df_above_70["grade"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculating the mean of training set grades
average_grade = y_train.mean()

# Assigning the mean grade as the prediction for all instances in the test set
test_constant_predictions = np.full_like(y_test, fill_value=average_grade)
train_constant_predictions = np.full_like(y_train, fill_value=average_grade)

# Evaluating the constant prediction
training_mae_6 = mean_absolute_error(y_train, train_constant_predictions)
training_mse_6 = mean_squared_error(y_train, train_constant_predictions)
test_mae_6 = mean_absolute_error(y_test, test_constant_predictions)
test_mse_6 = mean_squared_error(y_test, test_constant_predictions)
test_r2_score_3 = r2_score(y_test, test_constant_predictions)

# Printing evaluation metrics for the constant prediction using mean
print(f"Training MAE: {training_mae_6}")
print(f"Test MAE: {test_mae_6}")
print(f"Training MSE: {training_mse_6}")
print(f"Test MSE: {test_mse_6}")
print(f"Test R2 score: {test_r2_score_3}")


# Calculating the median of training set grades
median_grade = np.median(y_train)

# Assigning the median grade as the prediction for all instances in the test set
test_constant_predictions_median = np.full_like(y_test, fill_value=median_grade)
train_constant_predictions_median = np.full_like(y_train, fill_value=median_grade)

# Evaluating the constant prediction
training_mae_7 = mean_absolute_error(y_train, train_constant_predictions_median)
training_mse_7 = mean_squared_error(y_train, train_constant_predictions_median) 
test_mae_7 = mean_absolute_error(y_test, test_constant_predictions_median)
test_mse_7 = mean_squared_error(y_test, test_constant_predictions_median)
test_r2_score_4 = r2_score(y_test, test_constant_predictions_median)

# Printing evaluation metrics for the constant prediction using mean
print(f"Training MAE: {training_mae_7}")
print(f"Test MAE: {test_mae_7}")
print(f"Training MSE: {training_mse_7}")
print(f"Test MSE: {test_mse_7}")
print(f"Test R2 score: {test_r2_score_4}")


# # Evaluation

maes = {"Neural Network 1 (W2V vector is excluded)": {"Training": training_mae_1, "Test":  test_mae_1},
        "Neural Network 2 (W2V vector is included)": {"Training": training_mae_2, "Test": test_mae_2},
        "Decision Tree": {"Training": training_mae_3, "Test": test_mae_3},
        "Random Forest": {"Training": training_mae_4, "Test": test_mae_4},
        "XGBoost": {"Training": training_mae_5, "Test": test_mae_5},
        "Mean (Dummy Regressor)": {"Training": training_mae_6, "Test": test_mae_6},
        "Median (Dummy Regressor)": {"Training": training_mae_7, "Test": test_mae_7}}

mses = {"Neural Network 1 (W2V vector is excluded)": {"Training": training_mse_1, "Test":  test_mse_1},
        "Neural Network 2 (W2V vector is included)": {"Training": training_mse_2, "Test": test_mse_2},
        "Decision Tree": {"Training": training_mse_3, "Test": test_mse_3},
        "Random Forest": {"Training": training_mse_4, "Test": test_mse_4},
        "XGBoost": {"Training": training_mse_5, "Test": test_mse_5},
        "Mean (Dummy Regressor)": {"Training": training_mse_6, "Test": test_mse_6},
        "Median (Dummy Regressor)": {"Training": training_mse_7, "Test": test_mse_7}}


def plot_metrics(metrics, title):
    models = list(metrics.keys())
    training_errors = [metrics[model]['Training'] for model in models]
    test_errors = [metrics[model]['Test'] for model in models]

    # Setting the positions and width for the bars
    pos = np.arange(len(models))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(pos, training_errors, bar_width, label='Training', alpha=0.7)
    plt.bar(pos + bar_width, test_errors, bar_width, label='Test', alpha=0.7)

    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title(title)
    plt.xticks(pos + bar_width / 2, models, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.show()


plot_metrics(maes, "Mean Absolute Error (MAE) by Model")
plot_metrics(mses, "Mean Squared Error (MSE) by Model")


# Extracting "grade" and "cluster" for the plot
plot_data = clustered_temp_df_regular_extra_1[["grade", "cluster"]]

# Creating a new column for y-axis to represent clusters distinctly
plot_data["y_random"] = np.random.rand(len(plot_data))

# Plotting
plt.figure(figsize=(12, 6))
sns.scatterplot(x="grade", y="y_random", hue="cluster", data=plot_data, palette="bright", legend="full", s=100)  # s is the size of points
plt.title("Results of Clustering")
plt.xlabel("Grade")
plt.ylabel("")
plt.yticks([])  # Hiding y-axis ticks
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc=2)  # Moving the legend outside the plot
plt.show()


print("MAEs of our models: ")
pprint(maes)

print("MSEs of our models: ")
pprint(mses)


print("Model with the best MAE in training:")
mae_training_min = 100
model_name = ""
for model in maes.keys():
    if maes[model]["Training"] < mae_training_min:
        mae_training_min = maes[model]["Training"]
        model_name = model
    else:
        pass

print(model_name)

print("Model with the best MAE in test:")
mae_test_min = 100
model_name = ""
for model in maes.keys():
    if maes[model]["Test"] < mae_test_min:
        mae_test_min = maes[model]["Test"]
        model_name = model
    else:
        pass

print(model_name)

print("Model with the best MSE in training:")
mse_training_min = 100
model_name = ""
for model in mses.keys():
    if mses[model]["Training"] < mse_training_min:
        mse_training_min = mses[model]["Training"]
        model_name = model
    else:
        pass

print(model_name)

print("Model with the best MSE in test:")
mse_test_min = 100
model_name = ""
for model in mses.keys():
    if mses[model]["Test"] < mse_test_min:
        mse_test_min = mses[model]["Test"]
        model_name = model
    else:
        pass

print(model_name)

