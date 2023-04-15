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
        res = {}
        for q, d in d1.items():
            merged = {**d1[q], **d2[q]}
            res[q] = merged

        return res

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
                        self.qrel_dict[qID]["relevant"] = set()
                        self.qrel_dict[qID]["nonrelevant"] = set()
                    if score == 0:
                        self.qrel_dict[qID]["nonrelevant"].add(docID)# the empty dictionary will later contain the doc's features
                    else:
                        self.qrel_dict[qID]["relevant"].add(docID)
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
        sum_rel = 0
        sum_nonrel= 0
        sum_exp_nonrel = 0

        for qid, docs in self.qrel_dict.items():
            rel = docs["relevant"]
            sum_rel += len(rel)
            nonrel = docs["nonrelevant"]
            sum_nonrel += len(nonrel)

            dataset[qid] = {}
            dataset[qid]["relevant"] = rel

            for doc, score in self.es_scores[qid].items():
                if len(nonrel) < 1000 :
                    nonrel.add(doc)

            if len(nonrel) < 1000: # if still don't have enough docs, add some more irrelevant docs
                for doc, score in self.es_scores[59].items():
                    if len(nonrel) < 1000:
                        nonrel.add(doc)

            sum_exp_nonrel += len(nonrel)

            dataset[qid]["nonrelevant"] = nonrel

            print(len(rel), len(nonrel))

        print(sum_rel, sum_nonrel, sum_exp_nonrel)
        return dataset


    # creates csv file containing all query:document instances and their feature scores
    def init_feature_table(self):
        dataset = self.complete_data_set()

        with open("/Users/ellataira/Desktop/is4200/homework--6-ellataira/data/docs.csv", 'w', newline='') as opened:
            writer = csv.writer(opened)
            writer.writerow(["q:doc_id", "es", "okapi-tf", "tf-idf", "okapi-bm25","laplace", "jm", "label"])
            c = 0
            for qid, docs in dataset.items():
                r_nr = ["relevant", "nonrelevant"]
                for r in r_nr: # iterate over relevant and nonrelevant docs
                    if r == "relevant":
                        rel = 1
                    else:
                        rel = 0

                    for doc in list(docs[r]): # update features

                        es, ok, tf, bm, l, jm = self.try_get_scores(qid,doc)

                        writer.writerow([str(qid) + ":" + doc,
                                         es,
                                         ok,
                                         tf,
                                         bm,
                                         l,
                                         jm,
                                         rel])
                        c += 1

        opened.close()
        print(c)


    # try to access score of given qid:doc, or give score of 0
    def try_get_scores(self,qid, doc):
        # if key error, then irrelevant => == 0
        try:
            es = self.es_scores[qid][doc]
        except:
            es = 0
        try:
            ok = self.okapi_scores[qid][doc]
        except:
            ok = 0
        try:
            tf = self.tf_idf_scores[qid][doc]
        except:
            tf = 0
        try:
            bm = self.bm25_scores[qid][doc]
        except:
            bm = 0
        try:
            l = self.laplace_scores[qid][doc]
        except:
            l = 0
        try:
            jm = self.jm_scores[qid][doc]
        except:
            jm = 0

        return es, ok, tf, bm, l, jm


    # initializes csv matrix to a dataframe and normalizes scores
    def init_and_normalize_dataframe(self):
        df = pd.read_csv("/Users/ellataira/Desktop/is4200/homework--6-ellataira/data/docs.csv", index_col=0)
        rs = RobustScaler()
        columns = ["es", "okapi-tf", "tf-idf", "okapi-bm25", "laplace", "jm", "label"]
        df[columns] = rs.fit_transform(df[columns])

        return df


if __name__ == "__main__":
    p = preprocess()

