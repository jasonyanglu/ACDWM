from underbagging import *
from HoeffdingTree.hoeffdingtree import HoeffdingTree
from HoeffdingTree.core.attribute import Attribute
from HoeffdingTree.core.dataset import Dataset
from HoeffdingTree.core.instance import Instance
from scipy.stats.mstats import mquantiles
from queue import Queue
from sklearn.model_selection import train_test_split
import time

import abc


class OnlineBase:

    @abc.abstractmethod
    def update(self, data, label):
        pass


class DDM_OCI(OnlineBase):

    def __init__(self, decay_factor=0.9):

        self.decay_factor = decay_factor

        self.oob_model = OOB()
        self.pos_rec = 0
        self.neg_rec = 0
        self.stored_data = array([])
        self.stored_label = array([])
        self.warning = False
        self.p = array([])
        self.s = array([])
        self.ddm_n = 0
        self.train_count = 0

    def update(self, data, label):

        if self.train_count % 1000 == 0:
            print('Data ' + str(self.train_count))

        # store data since warning is issued
        if self.warning:
            self.stored_data = r_[self.stored_data.reshape(-1, data.size), data.reshape(1, -1)]
            self.stored_label = r_[self.stored_label, label]

        # update and predict
        pred = self.oob_model.update(data, label)
        self.ddm_n += 1
        self.train_count += 1

        # trace minority class recall and detect drift
        if label == 1:
            self.pos_rec = self.decay_factor * self.pos_rec + (1 - self.decay_factor) * (pred == label)
        else:
            self.neg_rec = self.decay_factor * self.neg_rec + (1 - self.decay_factor) * (pred == label)

        min_class = self.oob_model.get_minority_class()
        if min_class == -1:
            self.p = append(self.p, self.neg_rec)
        else:
            self.p = append(self.p, self.pos_rec)
        self.s = append(self.s, sqrt(self.p[-1] * (1 - self.p[-1]) / self.ddm_n))
        p_max = max(self.p)
        s_max = max(self.s)

        if self.p[-1] - self.s[-1] < p_max - 2 * s_max and self.warning == False:
            self.warning = True
            # print('Warning!')
        elif self.p[-1] - self.s[-1] < p_max - 3 * s_max and self.warning == True:
            # reset model
            # print('Drift detected!')
            self.oob_model = OOB()
            self.pos_rec = 0
            self.neg_rec = 0
            self.warning = False
            self.p = array([])
            self.s = array([])
            self.ddm_n = 0
            for i in range(self.stored_data.shape[0]):
                self.oob_model.update(self.stored_data[i], self.stored_label[i])
            self.stored_data = array([])
            self.stored_label = array([])

        return pred


class HLFR(OnlineBase):

    def __init__(self, decay_factor=0.9, warn_sig=0.01, detect_sig=0.0001, perm_sig=0.05, W=100, P=100):

        self.decay_factor = decay_factor
        self.warn_sig = warn_sig
        self.detect_sig = detect_sig
        self.perm_sig = perm_sig
        self.W = W
        self.P = P

        self.rate_name = ['tpr', 'tnr', 'ppv', 'npv']

        self.oob_model = OOB()
        self.stored_data = array([])
        self.stored_label = array([])
        self.warning = False
        self.R = {n: 0 for n in self.rate_name}
        self.C = ones([2, 2])
        self.window_data = Queue(2 * W)
        self.window_label = Queue(2 * W)
        self.pot_count = -1
        self.train_count = 0
        self.stored_decay = array([])

    def update(self, data, label):

        if self.train_count % 1000 == 0:
            print('Data ' + str(self.train_count))

        # store data since warning is issued
        if self.warning:
            self.stored_data = r_[self.stored_data.reshape(-1, data.size), data.reshape(1, -1)]
            self.stored_label = r_[self.stored_label, label]
        if self.window_data.full():
            self.window_data.get()
            self.window_label.get()
        self.window_data.put(data)
        self.window_label.put(label)

        # permutation test
        if self.pot_count != -1:
            pred = sign(self.oob_model.predict(data))

            if self.pot_count == self.window_data.qsize():
                window_data = array(self.window_data.queue)
                window_label = array(self.window_label.queue)
                if not self._permutation_test(window_data, window_label):
                    print('Level II detected at %d !' % self.train_count)
                    self.oob_model = OOB()

                for i in range(self.W):
                    self.oob_model.update(window_data[self.W + i], window_label[self.W + i])
                self.pot_count = -1
            else:
                self.pot_count += 1

        else:
            # update and predict
            pred = self.oob_model.update(data, label)

            # trace four rates and detect drift
            self.C[int(pred / 2 + 0.5), int(label / 2 + 0.5)] = self.C[int(pred / 2 + 0.5), int(label / 2 + 0.5)] + 1

            warn_bd = {n: 0 for n in self.rate_name}
            detect_bd = {n: 0 for n in self.rate_name}
            time_a = time.time()
            for rate in self.rate_name:
                if (rate == 'tpr' and label == 1) or (rate == 'tnr' and label == -1) or \
                        (rate == 'ppv' and pred == 1) or (rate == 'npv' and pred == -1):
                    self.R[rate] = self.decay_factor * self.R[rate] + (1 - self.decay_factor) * (pred == label)

                if rate in ['tpr', 'tnr']:
                    N = self.C[0, int(rate == 'tpr')] + self.C[1, int(rate == 'tpr')]
                    P = self.C[int(rate == 'tpr'), int(rate == 'tpr')] / N
                else:
                    N = self.C[int(rate == 'ppv'), 0] + self.C[int(rate == 'ppv'), 1]
                    P = self.C[int(rate == 'ppv'), int(rate == 'ppv')] / N

                warn_bd[rate] = self._bound_table(P, self.warn_sig, int(N))
                detect_bd[rate] = self._bound_table(P, self.detect_sig, int(N))

            if (sum([self.R[rate] > warn_bd[rate][1] for rate in self.rate_name]) > 0 or \
                sum([self.R[rate] < warn_bd[rate][0] for rate in self.rate_name]) > 0) and \
                    not self.warning:
                print('Warning at %d !' % self.train_count)
                self.warning = True
            elif sum([self.R[rate] > warn_bd[rate][1] for rate in self.rate_name]) == 0 and \
                    sum([self.R[rate] < warn_bd[rate][0] for rate in self.rate_name]) == 0 and \
                    self.warning:
                print('Warning cancel at %d !' % self.train_count)
                self.warning = False
                self.stored_data = array([])
                self.stored_label = array([])

            if (sum([self.R[rate] > detect_bd[rate][1] for rate in self.rate_name]) > 0 or \
                sum([self.R[rate] < detect_bd[rate][0] for rate in self.rate_name]) > 0) and \
                    len(self.stored_label) > 0:
                # reset model
                print('Level I detected at %d !' % self.train_count)
                self.oob_model = OOB()
                self.warning = False
                self.R = {n: 0 for n in self.rate_name}
                self.C = ones([2, 2])

                for i in range(self.stored_data.shape[0]):
                    self.oob_model.update(self.stored_data[i], self.stored_label[i])
                self.stored_data = array([])
                self.stored_label = array([])
                self.pot_count = 0

        self.train_count += 1

        return pred

    def _bound_table(self, P, alpha, N, MC=100):

        if len(self.stored_decay) < N:
            for i in range(len(self.stored_decay), N):
                self.stored_decay = append(self.stored_decay, self.decay_factor ** i)
        R = zeros(MC)
        for i in range(MC):
            bin_rand_num = random.binomial(1, P, N)
            R[i] = (1 - self.decay_factor) * sum(bin_rand_num * self.stored_decay[:N])

        bd = zeros(2)
        bd[0] = mquantiles(R, alpha)
        bd[1] = mquantiles(R, 1 - alpha)

        return bd

    def _permutation_test(self, data, label):

        model_ord = OOB()
        for i in range(self.W):
            model_ord.update(data[i], label[i])
        pred_ord = sign(model_ord.predict(data[self.W:]))
        loss_ord = sum(pred_ord != label[self.W:])

        loss_perm = zeros(self.P)
        # print('Permutation test')
        for p in range(self.P):
            try:
                X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                                    test_size=0.5, stratify=label)
            except ValueError:
                return False
            model_perm = OOB()
            for _ in range(self.W):
                model_perm.update(X_train[i], y_train[i])
            pred_perm = sign(model_ord.predict(X_test))
            loss_perm[p] = sum(pred_ord != y_test)

        test_value = (1 + sum(loss_ord < loss_perm)) / (self.P + 1)
        if test_value < self.perm_sig:
            return True
        else:
            return False


class PAUC_PH(OnlineBase):

    def __init__(self, window_size=1000, ph_delta=0.1, ph_lambda=100):

        self.window_size = window_size
        self.ph_delta = ph_delta
        self.ph_lambda = ph_lambda

        self.oob_model = OOB(prob=True)
        self.W_score = array([])
        self.W_label = array([])
        self.m = array([])
        self.auc = array([])

        self.train_count = 0
        self.crt_count = 0
        self.n = 0
        self.p = 0

    def update(self, data, label):

        if self.train_count % 1000 == 0:
            print('Data ' + str(self.train_count))

        # update and predict
        score = self.oob_model.update(data, label)
        self.train_count += 1
        self.crt_count += 1

        auc = self._prequential_auc(score, label)
        self.auc = r_[self.auc, auc]

        if self._ph_test(auc) == True:
            # print('Drift detected at %d !' % self.train_count)

            self.oob_model = OOB(prob=True)
            self.W_score = array([])
            self.W_label = array([])
            self.m = array([])
            self.auc = array([])

            self.crt_count = 0
            self.n = 0
            self.p = 0

        return sign(score)

    def _prequential_auc(self, score, label):

        if self.crt_count > self.window_size:
            del_idx = (self.crt_count - 1) % self.window_size
            self.W_score = delete(self.W_score, del_idx)
            if self.W_label[del_idx] == 1:
                self.p -= 1
            else:
                self.n -= 1
            self.W_label = delete(self.W_label, del_idx)

        self.W_score = r_[self.W_score, score]
        self.W_label = r_[self.W_label, label]

        if label == 1:
            self.p += 1
        else:
            self.n += 1

        sort_idx = argsort(-self.W_score)
        self.W_score = self.W_score[sort_idx]
        self.W_label = self.W_label[sort_idx]

        AUC = 0
        c = 0
        for i in range(self.W_score.size):
            if self.W_label[i] == 1:
                c += 1
            else:
                AUC += c
        if self.p * self.n != 0:
            return AUC / (self.p * self.n)
        else:
            return 0

    def _ph_test(self, auc):

        temp = (1 - self.auc) - mean(1 - self.auc) - self.ph_delta
        m_t = sum(temp[temp > 0])
        self.m = r_[self.m, m_t]
        if abs(m_t - min(self.m)) > self.ph_lambda:
            return True
        else:
            return False


class OOB():

    def __init__(self, T=11, theta=0.9, prob=False, silence=True):

        self.T = T
        self.theta = theta
        self.prob = prob
        self.silence = silence

        # Hoeffding tree ensemble init
        self.ensemble = list()
        for t in range(self.T):
            vfdt = HoeffdingTree()
            vfdt.set_grace_period(50)
            vfdt.set_hoeffding_tie_threshold(0.05)
            vfdt.set_split_confidence(0.0001)
            vfdt.set_minimum_fraction_of_weight_info_gain(0.01)
            self.ensemble.append(vfdt)

        self.train_count = 0
        self.w = array([0.5, 0.5])

    def _init_dataset(self, data, label):

        fea_num = data.size
        attributes = []
        for i in range(fea_num):
            attributes.append(Attribute(str(i), att_type='Numeric'))
        attributes.append(Attribute('Label', ['-1', '1'], att_type='Nominal'))

        self.dataset = Dataset(attributes, fea_num)

        inst_values = list(r_[data, label])
        inst_values[fea_num] = int(attributes[fea_num].index_of_value(str(int(label))))
        self.dataset.add(Instance(att_values=inst_values))

        for t in range(self.T):
            self.ensemble[t].build_classifier(self.dataset)

    def update(self, data, label):

        fea_num = data.size
        if self.train_count % 1000 == 0 and self.silence == False:
            print('Data ' + str(self.train_count))

        # format sample and predict
        if self.train_count == 0:
            pred = 0
        else:
            inst_values = list(r_[data, label])
            inst_values[fea_num] = int(self.dataset.attribute(index=fea_num).index_of_value(str(int(label))))
            new_instance = Instance(att_values=inst_values)
            new_instance.set_dataset(self.dataset)
            pred = self._predict(new_instance)

        # update prob
        self.w[0] = self.theta * self.w[0] + (1 - self.theta) * (label == -1)
        self.w[1] = self.theta * self.w[1] + (1 - self.theta) * (label == 1)

        # calculate sampling rate
        if label == 1 and self.w[1] < self.w[0]:
            sampling_rate = self.w[0] / self.w[1]
        elif label == -1 and self.w[1] > self.w[0]:
            sampling_rate = self.w[1] / self.w[0]
        else:
            sampling_rate = 1

        # incrementally train Hoeffding tree
        if self.train_count == 0:
            self._init_dataset(data, label)
        else:
            for t in range(self.T):
                K = random.poisson(sampling_rate)
                for _ in range(K):
                    self.ensemble[t].update_classifier(new_instance)

        self.train_count += 1

        if self.prob:
            return pred
        else:
            return sign(pred)

    def _predict(self, data):

        pred = zeros([self.T])
        for t in range(self.T):
            pred[t] = self.ensemble[t].distribution_for_instance(data)[1]
            pred[t] = (pred[t] - 0.5) * 2

        return mean(pred)

    def predict(self, data):

        if len(data.shape) == 1:
            data = data.reshape(1, data.size)
            fea_num = data.size
        data_num = data.shape[0]
        fea_num = data.shape[1]
        pred = zeros([self.T, data_num])

        for t in range(self.T):
            for i in range(data_num):
                inst_values = list(r_[data[i], 1])
                inst_values[fea_num] = int(self.dataset.attribute(index=fea_num).index_of_value(str(1)))
                new_instance = Instance(att_values=inst_values)
                new_instance.set_dataset(self.dataset)
                pred[t, i] = self.ensemble[t].distribution_for_instance(new_instance)[1]
                pred[t, i] = (pred[t, i] - 0.5) * 2

        return mean(pred, 0)

    def get_minority_class(self):

        if self.w[0] < self.w[1]:
            return -1
        else:
            return 1
