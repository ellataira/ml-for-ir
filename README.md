[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/TxfK4bqz)
# HW6

## `preprocess.py`

This file generates the matrix setup of query-document pairs and their features. The matrix consists of a query ID- document ID pair as the index column, 6 features, and the relevancy score. The 6 features are the 
scores from the following retrieval models: Elasticsearch builtin, Okapi-TF, TF-IDF, Okapi-BM25, Unigram LM with La Place smoothing, and Unigram LM with Jelinek-Mercer smoothing. 

To generate the matrix, it is first instantiated to contain only documents listed in the QREL file. Then, if there are less than 1000 non-relevant files, then additional non-relevant files are added to the matrix to total 1000 non-relevant documents.
After generating all query-document pairs, the feature columns of the matrix are filled in from scores calculated by the retrieval models. I previously ran all models to return 2000 relevant documents per query, and those scores are used to fill in the matrix. 

The feature matrix is normalized and converted to a dataframe that is then used in training. 

## `train.py`

This file uses a SKLearn Logistic Regression to train on the previously generated dataframe. 

To train, the query list is first randomly split into 5 testing queries and 20 training queries. This method, `get_train_test_queries`, ensures that each query is only tested on once. Then, I retrieve the partial dataframes assigned to each query set. 

The model first trains on the dataset from the 20 training queries. Then, the trained model predicts the probability of being relevant for query-document pairs in the testing dataframe. These x_test results are saved to a txt file. 
After, the model is run on the x_train set to predict relevance probabilities and again writes the results to a txt file. 

To cross-validate the trained model's results, I run the training procedure 5 times to ensure all 25 queries are tested and trained on. No query is tested on twice through the 5 iterations. Each time the file is run, 10 txt files are generated for test set and train set scores at `n` iterations (5 for x_test, 5 for x_train). 

## Results

I assessed the average precision of the training model using the `trec_eval.pl` script and given QREL document. Below are the scores for the `0x_test_res[].txt` and `0x_train_res[].txt` files. 

Since the queries are randomly split into sets of 20 and 5 each time the program is run, the average precision will vary upon each trial, but the below scores are an example of a full execution. 

### Average Precision

| Iteration    | x_test  | x_train |
|--------------|---------|-----|
| 1            | 0.5001  |   0.3142 |
| 2            | 0.4034  |   0.3278 |
| 3            | 0.2424  |  0.3639 |
| 4            | 0.3047  | 0.3636 |
| 5            | 0.2459  | 0.3609 |
| **average:** | 0.3393  |0.34608 |

