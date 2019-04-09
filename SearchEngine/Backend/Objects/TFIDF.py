import numpy as np


class TFIDF:
    def __init__(self):
        self.term_id = []
        self.term_frequency_list = []
        self.tfidfs_norm = []

    def add_tidf(self, term_id, term_frequency, term_idf):
        self.term_id.append(term_id)
        # self.tfidfs.append(term_idf * term_frequency)
        self.term_frequency_list.append(term_frequency)

    def apply_normalization(self):
        tfidfnp = np.asarray(self.term_frequency_list)
        tfidfnp = np.power(tfidfnp, 2)
        doc_length = np.sqrt(np.sum(tfidfnp))
        self.tfidfs_norm = [tfidf / doc_length for tfidf in self.term_frequency_list]

    def get_tfidf_of_term_id(self, term_id):
        try:
            indx = self.term_id.index(term_id)
        except Exception:
            return None
        return self.term_frequency_list[indx]
        # for n, id in enumerate(self.term_id):
        #     if term_id == id:
        #         return self.tfidfs[n]
        # return None

    def get_tfidf_norm_of_term_id(self, term_id):
        try:
            indx = self.term_id.index(term_id)
        except Exception:
            return None
        return self.tfidfs_norm[indx]
        # for n, id in enumerate(self.term_id):
        #     if term_id == id:
        #         return self.tfidfs_norm[n]
        # return None
