import csv

import pandas as pd
from sklearn.preprocessing import RobustScaler
import query_execution

class preprocess:

    def __init__(self):
        self.qrel_dict = {}
        self.read_qrel("/Users/ellataira/Desktop/is4200/homework--6-ellataira/data/qrels.adhoc.51-100.AP89.txt")
        self.es_scores = self.merge_scores("/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/es_builtin.txt",
                                           "/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/qrel_es_builtin.txt")
        self.okapi_scores =  self.merge_scores("/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/okapi_tf.txt",
                                               "/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/qrel_okapi_tf.txt")
        self.tf_idf_scores =  self.merge_scores("/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/tf_idf.txt",
                                                "/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/qrel_tf_idf.txt")
        self.bm25_scores =  self.merge_scores("/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/okapi_bm25.txt",
                                              "/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/qrel_okapi_bm25.txt")
        self.laplace_scores =  self.merge_scores("/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/uni_lm_laplace.txt",
                                                 "/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/qrel_uni_lm_laplace.txt")
        self.jm_scores =  self.merge_scores("/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/uni_lm_jm.txt",
                                            "/Users/ellataira/Desktop/is4200/homework--6-ellataira/Results/qrel_uni_lm_jm.txt")
        # dictionary of  { qid : { {relevant : {doc: {features}} } , {nonrelevant: {doc: {features}} } } }
        self.init_feature_table()
        self.n_dataframe = self.init_and_normalize_dataframe()

    # merge qrel scores and es search scores into one master dict to pull from for features
    def merge_scores(self, search_res, qrel_res):
        d1 = self.read_scores(search_res)
        d2 = self.read_scores(qrel_res)

        d1.update(d2)

        return d1

    # reads given qrel file to create dictionary that maps query id to its relevant and nonrelevant documents
    def read_qrel(self, filename):
        qs = query_execution.process_all_queries("/Users/ellataira/Desktop/is4200/homework--6-ellataira/data/new_queries.txt").keys()
        with open(filename, 'rb') as opened:
            for line in opened:
                split_line = line.split()
                qID, docID, score = int(split_line[0].decode()), split_line[2].decode(), float(split_line[3].decode())
                if str(qID) in qs: # only include 25 queries from qrel
                    if qID not in self.qrel_dict.keys(): # add query to dict if not present
                        self.qrel_dict[qID] = {}
                        self.qrel_dict[qID]["relevant"] = {}
                        self.qrel_dict[qID]["nonrelevant"] = {}
                    if score == 0:
                        self.qrel_dict[qID]["nonrelevant"][docID] = {} # the empty dictionary will later contain the doc's features
                    else:
                        self.qrel_dict[qID]["relevant"][docID] = {}
        opened.close()

    # reads score output file of 2000 docs into dictionary
    def read_scores(self, to_add_filepath):
        scored_docs_dict = {} # map qid : docid : score
        # read in ES search results bc best ranking model
        with open(to_add_filepath, "rb") as opened:
            for line in opened:
                split_line = line.split()
                # query_id Q0 doc_id  rank score Exp
                qid, docid, score = int(split_line[0].decode()), split_line[2].decode(), float(split_line[4].decode()) # int, string, float
                if qid not in scored_docs_dict.keys():
                    scored_docs_dict[qid] = {}
                scored_docs_dict[qid][docid] = score
        opened.close()

        return scored_docs_dict

    # completes dataset so there are 1000 nonrelevant docs
    def complete_data_set(self):
        dataset = {}

        for qid, docs in self.qrel_dict.items():
            rel = docs["relevant"]
            nonrel = docs["nonrelevant"]

            dataset[qid] = {}
            dataset[qid]["relevant"] = rel

            for doc, score in self.es_scores[qid].items():
                if len(nonrel) < 1000 :
                    if doc not in rel and doc not in nonrel:
                        nonrel[doc] = {}
                else:
                    break

            dataset[qid]["nonrelevant"] = nonrel


        return dataset


    def init_feature_table(self):
        dataset = self.complete_data_set()

        with open("/Users/ellataira/Desktop/is4200/homework--6-ellataira/data/docs.csv", 'w', newline='') as opened:
            writer = csv.writer(opened)
            writer.writerow(["q:doc_id", "es", "okapi-tf", "tf-idf", "okapi-bm25","laplace", "jm", "label"])

            for qid, docs in dataset.items():
                r_nr = ["relevant", "nonrelevant"]
                for r in r_nr: # iterate over relevant and nonrelevant docs
                    if r == "relevant":
                        rel = 1
                    else:
                        rel = 0

                    for doc, features in docs[r].items(): # update features
                        writer.writerow([str(qid) + ":" + doc,
                                         self.es_scores[qid][doc],
                                         self.okapi_scores[qid][doc],
                                         self.tf_idf_scores[qid][doc],
                                         self.bm25_scores[qid][doc],
                                         self.laplace_scores[qid][doc],
                                         self.jm_scores[qid][doc],
                                         rel])

        opened.close()


    def init_and_normalize_dataframe(self):
        df = pd.read_csv("/Users/ellataira/Desktop/is4200/homework--6-ellataira/data/docs.csv", index_col=0)
        rs = RobustScaler()
        columns = ["es", "okapi-tf", "tf-idf", "okapi-bm25","laplace", "jm", "label"]
        df[columns] = rs.fit_transform(df[columns])

        return df


if __name__ == "__main__":
    p = preprocess()
