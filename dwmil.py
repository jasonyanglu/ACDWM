# Implement DWMIL
# By Yang Lu, 25/01/2018

from numpy import *
from subunderbagging import *
from underbagging import *
from sklearn.metrics import f1_score
from check_measure import *
from chunk_based_methods import ChunkBase
from sklearn.model_selection import StratifiedKFold
import pdb


class DWMIL(ChunkBase):

    def __init__(self, data_num, chunk_size=1000, theta=0.1, err_func='gm', r=1):
        ChunkBase.__init__(self)

        self.data_num = data_num
        self.chunk_size = chunk_size
        self.theta = theta
        self.err_func = err_func
        self.r = r

        self.ensemble_size_record = array([])

    def _update_chunk(self, data, label):
        model = UnderBagging(r=self.r, auto_T=True)
        model.train(data, label)
        self.ensemble.append(model)
        self.chunk_count += 1
        self.w = append(self.w, 1)
        all_pred = sign(self._predict_base(data))

        if self.chunk_count > 1:
            pred = dot(all_pred[:, :-1], self.w[:-1])
        else:
            pred = zeros_like(label)

        pred = sign(pred)
        err = self.calculate_err(all_pred, label)
        self.w = (1 - err) * self.w

        remove_idx = nonzero(self.w < self.theta)[0]
        if len(remove_idx) != 0:
            for index in sorted(remove_idx, reverse=True):
                del self.ensemble[index]
            self.w = delete(self.w, remove_idx)
            self.chunk_count -= remove_idx.size

        self.ensemble_size_record = r_[self.ensemble_size_record, len(self.ensemble)]

        return pred


