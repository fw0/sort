import numpy as np

class model(object):

    def add_placeholders(self)
        """
        returns: added placeholders as a tuple
        """
        raise NotImplementedError

    def add_embedding(self, *placeholders):
        """
        returns: embeddings u_ts, the context for encoder
        """
        raise NotImplementedError

    def add_sequence_model(self, u_ts):
        """
        returns: c_ts, the context for decoder
        """
        raise NotImplementedError
