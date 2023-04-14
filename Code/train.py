import random

import pandas as pd
from sklearn.linear_model import LogisticRegression

import query_execution
from preprocess import preprocess

SIZE = 1000 # save the top 1000 docs
class ML:

    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        self.queries = query_execution.process_all_queries("/Users/ellataira/Desktop/is4200/"
                                                                          "homework--6-ellataira/data/new_queries.txt")
        self.tested_queries = []

    def get_train_test_queries(self): # TODO is it ok that some queries are trained on multiple times? i feel like no...
        test_pool = list(self.queries.keys()) # only need query numbers, not contents

        for q in self.tested_queries: # update pool of queries to be tested (aka set of 5)
            test_pool.remove(q)

        test = random.sample(test_pool, 5)
        self.tested_queries.extend(test)

        train = set(self.queries.keys()) - set(test)

        return test, list(train)

    def get_partial_dataset(self, queries):
        frames = []

        for id, q in enumerate(queries):
            partial = self.dataframe[self.dataframe.index.str[0:2] == q]
            frames.append(partial)

        df_res = pd.concat(frames)
        return df_res

    def train(self):
        # iterate to train and test on all queries (cross-validation)
        for i in range(5):
            test, train = ml.get_train_test_queries()

            print("iter:" , i)
            print("queries:")
            print(test)
            print(train)

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


            # 1. train model on training set
            lr = LogisticRegression(max_iter=1000, solver='liblinear', C=0.01, penalty='l1')
            lr.fit(x_train, y_train)

            # score test set using trained model and save results
            test_res = lr.predict_proba(x_test)[:, 1] # [:,1] to only prob of being 1 # TODO WHAT DOES IT MEAN THAT THESE ARE ALL THE SAME VLAUES--  but that only happens sometimes
            self.write_result(test_res, i, test_df, "x_test_res")

            # score training set using trained model and save results
            train_res = lr.predict_proba(x_train)[:,1]
            self.write_result(train_res, i, train_df, "x_train_res")


    def write_result(self, probs, iter, test_df, filename):
        # need to sort probabilities by query in descending order while maintaining prob:data pairs
        sorted_list = {} # maps qid : [ (docid , prob) ]
        for row_id, p in enumerate(probs):
            qid_did = test_df.iloc[row_id].name
            qid = int(qid_did[0:2])
            did = qid_did[3::]
            if qid not in sorted_list.keys():
                sorted_list[qid] = []
            sorted_list[qid].append((did, p))

        # sort each query's docs by descending probability
        for qid, docs in sorted_list.items():
            s_docs = sorted(docs, key=lambda item: float(item[1]), reverse=True)
            sorted_list[qid] = s_docs[0:SIZE]

        # write to file
        with open("/Users/ellataira/Desktop/is4200/homework--6-ellataira"
                  "/Results/" + filename + str(iter)+ ".txt", "w") as opened:
            for qid, docs in sorted_list.items():
                count = 1
                for docid, prob in docs:
                    opened.write(str(qid) + ' Q0 ' + str(docid) + ' ' + str(count) + ' ' + str(prob) + ' Exp\n')
                    count += 1

        opened.close()


if __name__ == "__main__" :
    p = preprocess()
    ml = ML(p.n_dataframe)
    ml.train()

    # print(df.head())
    #
    # for i in range(5):
    #     print(ml.get_train_test_queries())
