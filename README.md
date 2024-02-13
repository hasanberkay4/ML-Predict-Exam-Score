# Overview of the Repository

This repository encompasses the comprehensive implementation of our machine learning project, focusing on enhancing the understanding of homework grades based on ChatGPT prompts. Below is a succinct breakdown of the main script and code components embedded within the repository.

### Data Preprocessing
The initial phase involves data preprocessing, including NLTK resource download and HTML text cleaning, extracting, and storing preprocessed conversations. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/558bb9997b7da86e6e039f5162f6c35964468c12/notebook.py#L51C1-L114C20).

### Matching Prompts with Questions 
In this step, user prompts are matched with their respective codes, preprocess a set of questions, and prepare them for subsequent analysis. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L117C1-L209C1).

### TF-IDF Vectorization 
The subsequent stage involves TF-IDF vectorization on user prompts and questions, calculating cosine similarity, and organizing results into a dataframe (question_mapping_scores). The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L211C1-L269C1).

### Outlier Treatment
To ensure data integrity, the code identifies and handles outliers by filtering entries based on average distances and a specified threshold. The codes of dropped entries are printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L271C1-L304C42).

### Word2Vec Vectorization
Utilizing Word2Vec, this step creates embeddings for prompts and questions, calculates cosine similarity scores, and generates individual Word2Vec models for each student. The resulting dataframe (code2word2vec) is printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L345C1-L401C1).

### Normalization
The normalization phase involves reading and preprocessing scores from a CSV file, removing grades received by outlier students, and performing Min-Max scaling on the grades. The resulting dataframe (normalized_scores) is printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L403C1-L442C25).

### Feature Engineering
Feature engineering includes sentiment analysis, keyword counting, and context detection for top-scoring students. Various scenarios are considered, and merged dataframes are created for regular scores, separate Word2Vec models, normalized scores, and separate Word2Vec models with normalized scores. The sorted dataframes are printed for reference. The code can be accessed [here](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L445C1-L634C62).

### Model Training
Finally, model training is initiated, encompassing Neural Networks, Clustering, Decision Tree, RandomForestRegressor, and XGBoost. The models' Mean Squared Error (MSE) scores are evaluated to identify the optimal one.

# Methodology

The methodology employed in this project encompasses a systematic exploration of five distinct machine learning algorithms to discern the most accurate approach. To evaluate the impact of outlier treatment, various experiments were conducted, including dropping detected outliers, utilizing data imputation with KNN, and training models on both approaches.

The model building process commenced with a [Decision Tree](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L780C1-L843C37). Utilizing scikit-learn in Python, a regressor was instantiated with a maximum depth of 10 and squared error as the criterion. The model underwent training on the specified data, followed by hyperparameter tuning through grid search. The resulting decision tree was visualized, and predictions were evaluated based on Mean Squared Error (MSE) and R2 score.

The approach extended to [RandomForestRegressor](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L846C1-L898C33), configuring parameters such as the number of estimators, maximum depth, and minimum samples split. A grid search optimized the model, and predictions on the test set were assessed through MSE. This process aimed to enhance the accuracy of regression predictions.

Leveraging the xgboost library, an [XGBoost](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L901C1-L1020C1) approach was implemented with hyperparameter tuning via GridSearchCV. The model's performance was evaluated using metrics like Mean Absolute Error (MAE), MSE, and R-squared, optimizing for accurate regression predictions.

As part of feature engineering, [Clustering](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L743C1-L777C41) (Figure 1) using the K-Means algorithm was explored. The Elbow Method guided the determination of the optimal number of clusters. K-Means clustering with the chosen number of clusters (num_clusters = 6) identified patterns within the data, and the resulting dataframe was saved for further analysis. 

![Figure 1: Visualization of Elbow Method for Optimal k](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/elbow_method.png?raw=true)
*Figure 1: Visualization of Elbow Method for Optimal k*

Two [Neural Network models](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L661C1-L741C1) were constructed, model_NN_1 utilized the normalized feature set, and model_NN_2 used the normalized feature set with the addition of Word2Vec vectors. Both models consisted of an input layer with 128 neurons, a hidden layer with 64 neurons, a dropout layer (0.7) to prevent overfitting, and an output layer with a linear activation function. Models were compiled using the Adam optimizer with a learning rate of 0.01 and mean squared error loss. Both models ran for 500 epochs, which seemed to ensure convergence. Both models were evaluated for test accuracies to explore the predictive capabilities of the dataset using different feature sets. 

This methodological diversity allowed for a comprehensive evaluation of the strengths and weaknesses of each approach, leading to an informed selection of the most effective model for accurate regression predictions on the given data.

Due to the distribution of the provided data, the team also worked on 2 [dummy classifiers](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/886d74f408f46c44deaed4f31264850ebf1abcd7/notebook.py#L1022C1-L1072C43) just to display the outcomes. The first model was designed to predict grades for instances with scores above 70. It used a simple strategy by calculating the mean of the training set grades and assigning this mean grade as the prediction for all instances in the test set. The second model used the median of the training set grades for prediction. This model assigned the median grade as the constant prediction for all instances in the test set.

These dummy models served as simplistic benchmarks, highlighting the importance of developing predictive models that outperform such elementary strategies. These dummy classifiers provided a reference point for the performance of more sophisticated models developed during the project.

# Results

## Clustering:
The clustering plot without (Figure 2) and with kNN imputation (Figure 3) displays the distribution of grades in a 2-dimensional space. Each dot represents a conversation, color-coded by the cluster it belongs to. There appears to be a degree of separation between clusters, suggesting that the conversations have some underlying patterns that are related to the grades received. The clustering does not show distinct groupings by grades in either case, indicating that the relationship between conversation features and grades is complex.

![Figure 2: Results of clustering without kNN imputation (outliers and students with empty prompts are dropped)](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/clustering.png?raw=true)
*Figure 2: Results of clustering without kNN imputation (outliers and students with empty prompts are dropped)*

![Figure 3: Results of clustering with kNN imputation](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/clustering_knn.png?raw=true)
*Figure 3: Results of clustering with kNN imputation*

## MAE and MSE Results:
The MAE and MSE plots without (Figure 4 and 5 respectively) and with kNN imputation (Figure 6 and 7 respectively) compare the performance of different models. These include two neural network configurations (with and without word2vec vectors), Decision Tree, Random Forest, XGBoost, and two dummy classifiers (mean and median strategies).

![Figure 4: MAEs without kNN imputation (outliers and students with empty prompts are dropped)](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mae.png?raw=true)
*Figure 4: MAEs without kNN imputation (outliers and students with empty prompts are dropped)*

![Figure 5: MSEs without kNN imputation (outliers and students with empty prompts are dropped)](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mse.png?raw=true)
*Figure 5: MSEs without kNN imputation (outliers and students with empty prompts are dropped)*

![Figure 6: MAEs with kNN imputation](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mae_knn.png?raw=true)
*Figure 6: MAEs with kNN imputation*

![Figure 7: MSEs with kNN imputation](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/mse_knn.png?raw=true)
*Figure 7: MSEs with kNN imputation*

### Neural Networks:
Before kNN imputation, neural network models demonstrated solid performance. The neural network with word2vec (W2V) vectors included (Neural Network 2) displayed marginally better performance compared to the one without (Neural Network 1), as seen in both MAE and MSE metrics. The error rates were low, suggesting these models captured the underlying patterns in the data quite effectively. Following kNN imputation, both neural networks experienced a slight increase in error rates, with Neural Network 2 (W2V included) maintaining its edge as the better model of the two, although the gap between them narrowed slightly.

### Decision Tree:
The Decision Tree shows a notable disparity between training and test errors, with a significantly lower error on the training set. This is indicative of overfitting, where the model captures noise in the training data that does not generalize well to unseen data.

Post-kNN imputation, the Decision Tree's performance deteriorates further on the test set, suggesting that the imputation might have introduced complexities that the model is overfitting to even more.

[Figure 8](https://github.com/dmuratli/CS412-Fall2023-Group-Project/blob/main/Figures/hw.pdf) displays the decision tree in a graphic format.

### Random Forest:
The Random Forest model, which is an ensemble of decision trees, typically reduces overfitting by averaging the results of individual trees. This is observed in the less pronounced gap between training and test errors compared to the single Decision Tree model.

After kNN imputation, the Random Forest's errors increased, but it still maintained a more stable performance relative to the Decision Tree, which points to its robustness despite the potentially noisy imputed data.

### XGBoost:
XGBoost, employing a sequential tree-building strategy, stood out as the best performing model pre-imputation, with the lowest error rates on both the training and test sets, indicating strong predictive power. However, it also showed a substantial overfitting, with a larger difference between training and test errors. Following kNN imputation, XGBoost's error rates increased, but it continued to be the best performer amongst all models, even though the overfitting issue persisted.

### Dummy Regressors:
Remarkably, the dummy regressors—simple models that predict the mean (Mean Dummy Regressor) or median (Median Dummy Regressor) value of the training set grades—exhibited a level of performance that was competitive with the more complex models. Before kNN imputation, their MAE and MSE values were in the same range as those of the sophisticated models, which is surprising given that dummy regressors do not actually "learn" from the data. This suggests that the other models may not be fully capturing the underlying patterns and could be overfitting to the noise within the training data.

After kNN imputation, the performance of the dummy regressors relative to the other models changed slightly, but they still remained competitive. This is particularly noteworthy in the context of the test set, where maintaining performance post-imputation is challenging. The dummy regressors' ability to perform on par with or even better than some of the more complex models raises important considerations about the nature of the dataset and the potential for simpler models to provide a baseline that is difficult to outperform. This could indicate that either the features used in the more complex models are not sufficiently informative or that the complexity of these models is not justified given the data at hand.

## Conclusion
In conclusion, the prediction of homework scores based on ChatGPT conversations reveals that XGBoost outperforms all other models in terms of error metrics, both before and after kNN imputation, despite showing signs of significant overfitting. The neural network models rank as the second best, with the one including W2V vectors marginally outperforming the one without. The increased error rates post-imputation across all models suggest that the kNN method may not be the most suitable for this dataset, and alternative imputation techniques could be considered to enhance model performance.

# Team Contributions
The team collaboratively discussed the project's initial steps and distributed tasks during meetings. Working together, the team focused on preprocessing, matching prompts with questions, and TF-IDF vectorization, utilizing code provided by the CS412 team. Hasan and Sude worked on the Outlier Treatment, exploring alternative approaches such as k-nn filling and dropping. Deniz took charge of Word2Vec vectorization and Normalization using MinMaxScaling. Melis was responsible for Feature Engineering, experimenting with different wordings for feature extraction, including GPT responses such as apologies or assessing if the student provided context to GPT.  Betül explored alternative approaches for feature selection. Following these tasks, the team gathered to discuss further steps, implementing multiple models for comparison and aiming for a more accurate method. Hasan and Betül collaborated on building Decision Tree, Random Forest, and XGBoost models, assessing MSE results. Deniz constructed a Neural Network (NN) model and teamed up with Sude on Clustering. Melis helped with the visualization of outcomes of model predictions, closely working with Deniz. Lastly, all team members cooperated in editing the repository and code flow, as well as contributing to the readme file.
