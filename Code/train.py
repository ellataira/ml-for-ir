import random

import query_execution

class ML:

    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        self.queries = query_execution.process_all_queries("/Users/ellataira/Desktop/is4200/"
                                                                          "homework--6-ellataira/data/new_queries.txt")
        self.tested_queries = []

    def get_train_test_queries(self):
        test_pool = list(self.queries.keys()) # only need query numbers, not contents
        print(self.tested_queries)

        for q in self.tested_queries: # update pool of queries to be tested (aka set of 5)
            test_pool.remove(q)

        test = random.sample(test_pool, 5)
        self.tested_queries.extend(test)

        print(self.tested_queries)
        train = set(self.queries.keys()) - set(test)

        return test, list(train)

if __name__ == "__main__" :
    ml = ML()

    for i in range(5):
        print(ml.get_train_test_queries())
