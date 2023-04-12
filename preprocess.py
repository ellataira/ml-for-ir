import pandas as pd
from sklearn.preprocessing import RobustScaler

class preprocess:

    def __init__(self):
        self.qrel_dict = {}
        self.es_scores = self.read_scores() # TODO ADD FILE NAMES
        self.okapi_scores =  self.read_scores()
        self.tf_idf_scores =  self.read_scores()
        self.bm25_scores =  self.read_scores()
        self.laplace_scores =  self.read_scores()
        self.jm_scores =  self.read_scores()
        # dictionary of  { qid : { {relevant : {doc: {features}} } , {nonrelevant: {doc: {features}} } } }
        self.data_set = self.complete_data_set()

    # reads given qrel file to create dictionary that maps query id to its relevant and nonrelevant documents
    def read_qrel(self, filename):
        with open(filename, 'rb') as opened:
            for line in opened:
                split_line = line.split()
                qID, assessorID, docID, score = split_line
                score = int(score)
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
                qid, docid, score = split_line[0], split_line[2], split_line[4]
                if qid not in scored_docs_dict.keys():
                    scored_docs_dict[qid] = []
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


    def update_features(self):
        for qid, docs in self.data_set.items():
            r_nr = ["relevant", "nonrelvant"]
            for r in r_nr: # iterate over relevant and nonrelevant docs
                for doc, features in docs[r].items(): # update features
                    f_dict = {"es": self.es_scores[qid][doc],
                              "okapi" : self.okapi_scores[qid][doc],
                              "tf" : self.tf_idf_scores[qid][doc],
                              "bm25": self.bm25_scores[qid][doc],
                              "lp": self.laplace_scores[qid][doc],
                              "jm": self.jm_scores[qid][doc]}
                    self.data_set[qid][r][doc].update(f_dict)


    def init_and_normalize_dataframe(self):
        df = pd.DataFrame.from_dict(self.data_set, index_col=0)
        rs = RobustScaler()
        columns = ["es", "okapi", "tf", "bm25", "lp", "jm"]
        df[columns] = rs.fit_transform(df[columns])

