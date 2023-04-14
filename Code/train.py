import random

import pandas as pd
from sklearn.linear_model import LogisticRegression

import query_execution
from preprocess import preprocess

class ML:

    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        self.queries = query_execution.process_all_queries("/Users/ellataira/Desktop/is4200/"
                                                                          "homework--6-ellataira/data/new_queries.txt")
        self.tested_queries = []

    def get_train_test_queries(self):
        test_pool = list(self.queries.keys()) # only need query numbers, not contents

        for q in self.tested_queries: # update pool of queries to be tested (aka set of 5)
            test_pool.remove(q)

        test = random.sample(test_pool, 5)
        self.tested_queries.extend(test)

        train = set(self.queries.keys()) - set(test)

        return test, list(train)

    def get_partial_dataset(self, queries):
        frames=  []

        for id, q in enumerate(queries):
            if id == 0 :
                partial = self.dataframe[self.dataframe.index.str[0:2] == q]
                frames.append(partial) # TODO HOW TO MERGE DATAFRAMES

        df_res = pd.concat(frames)
        return df_res

    def train(self):
        test, train = ml.get_train_test_queries()

        test_df = ml.get_partial_dataset(test)
        train_df = ml.get_partial_dataset(train)

        # x_train = features of training
        # y_train = target of training
        # x_test = features of testing
        # y_test = target of training

        x_train, y_train, x_test, y_test = train_df.iloc[:,:-1], train_df.iloc[:,-1], \
                                           test_df.iloc[:,:-1], test_df.iloc[:,-1]

        # print(x_train)
        # print(y_train)
        # print(x_test)
        # print(y_test)

        lr = LogisticRegression(max_iter=1000, solver='liblinear', C=0.01, penalty='l1')
        lr.fit(x_train, y_train)

        res = lr.predict_proba(x_test)[:,1] # [:,1] to only prob of being 1 # TODO WHAT DOES IT MEAN THAT THESE ARE ALL THE SAME VLAUES--  but that only happens sometimes
        print(res)




if __name__ == "__main__" :
    p = preprocess()
    ml = ML(p.n_dataframe)
    ml.train()

    # print(df.head())
    #
    # for i in range(5):
    #     print(ml.get_train_test_queries())
