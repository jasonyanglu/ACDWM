# Compare ACDWM with other online learning methods
# Example: python main_compare.py sea_abrupt.npz

from numpy import *
from chunk_size_select import *
from dwmil import *
from chunk_based_methods import *
from online_methods import *
from check_measure import *
import matplotlib.pyplot as plt
import sys

data_name = sys.argv[1]

load_data = load('data/' + data_name)
data = load_data['data']
label = load_data['label']
reset_pos = load_data['reset_pos'].astype(int)

data_num = data.shape[0]
chunk_size = 1000

run_num = 1

pq_result_ub = [{} for _ in range(run_num)]
pq_result_rea = [{} for _ in range(run_num)]
pq_result_dwmil = [{} for _ in range(run_num)]
pq_result_acdwm = [{} for _ in range(run_num)]
pq_result_learnpp_nie = [{} for _ in range(run_num)]
pq_result_dfgw_is = [{} for _ in range(run_num)]
pq_result_oob = [{} for _ in range(run_num)]
pq_result_ddm_oci = [{} for _ in range(run_num)]
pq_result_hlfr = [{} for _ in range(run_num)]
pq_result_pauc_ph = [{} for _ in range(run_num)]

for run_i in range(run_num):

    acss = ChunkSizeSelect()

    model_ub = UB(data_num=data_num, chunk_size=chunk_size)
    model_rea = REA(data_num=data_num, chunk_size=chunk_size)
    model_dwmil = DWMIL(data_num=data_num, chunk_size=chunk_size)
    model_acdwm = DWMIL(data_num=data_num, chunk_size=0)
    model_learnpp_nie = LearnppNIE(data_num=data_num, chunk_size=chunk_size)
    model_dfgw_is = DFGWIS(fea_num=data.shape[1], data_num=data_num, chunk_size=chunk_size)
    model_oob = OOB(silence=False)
    model_ddm_oci = DDM_OCI()
    model_hlfr = HLFR()
    model_pauc_ph = PAUC_PH()

    pred_ub = array([])
    pred_rea = array([])
    pred_dwmil = array([])
    pred_acdwm = array([])
    pred_learnpp_nie = array([])
    pred_dfgw_is = array([])
    pred_oob = array([])
    pred_ddm_oci = array([])
    pred_hlfr = array([])
    pred_pauc_ph = array([])

    print('Round ' + str(run_i))
    for i in range(data_num):

        pred_ub = append(pred_ub, model_ub.update(data[i], label[i]))
        pred_rea = append(pred_rea, model_rea.update(data[i], label[i]))

        pred_dwmil = append(pred_dwmil, model_dwmil.update(data[i], label[i]))
        pred_learnpp_nie = append(pred_learnpp_nie, model_learnpp_nie.update(data[i], label[i]))
        pred_dfgw_is = append(pred_dfgw_is, model_dfgw_is.update(data[i], label[i]))

        pred_oob = append(pred_oob, model_oob.update(data[i], label[i]))
        pred_ddm_oci = append(pred_ddm_oci, model_ddm_oci.update(data[i], label[i]))
        pred_hlfr = append(pred_hlfr, model_hlfr.update(data[i], label[i]))
        pred_pauc_ph = append(pred_pauc_ph, model_pauc_ph.update(data[i], label[i]))

        # acdwm
        acss.update(data[i], label[i])
        if i == data_num - 1:
            chunk_data, chunk_label = acss.get_chunk_2()
            pred_acdwm = append(pred_acdwm, model_acdwm.predict(chunk_data))
        elif acss.get_enough() == 1:
            chunk_data, chunk_label = acss.get_chunk()
            pred_acdwm = append(pred_acdwm, model_acdwm.update_chunk(chunk_data, chunk_label))

    pq_result_ub[run_i] = prequential_measure(pred_ub, label, reset_pos)
    pq_result_rea[run_i] = prequential_measure(pred_rea, label, reset_pos)
    pq_result_dwmil[run_i] = prequential_measure(pred_dwmil, label, reset_pos)
    pq_result_acdwm[run_i] = prequential_measure(pred_acdwm, label, reset_pos)
    pq_result_learnpp_nie[run_i] = prequential_measure(pred_learnpp_nie, label, reset_pos)
    pq_result_dfgw_is[run_i] = prequential_measure(pred_dfgw_is, label, reset_pos)
    pq_result_oob[run_i] = prequential_measure(pred_oob, label, reset_pos)
    pq_result_ddm_oci[run_i] = prequential_measure(pred_ddm_oci, label, reset_pos)
    pq_result_hlfr[run_i] = prequential_measure(pred_hlfr, label, reset_pos)
    pq_result_pauc_ph[run_i] = prequential_measure(pred_pauc_ph, label, reset_pos)


print('ub: %f' % mean([pq_result_ub[i]['gm'][-1] for i in range(run_num)]))
print('rea: %f' % mean([pq_result_rea[i]['gm'][-1] for i in range(run_num)]))
print('learnpp_nie: %f' % mean([pq_result_learnpp_nie[i]['gm'][-1] for i in range(run_num)]))
print('dfgw_is: %f' % mean([pq_result_dfgw_is[i]['gm'][-1] for i in range(run_num)]))
print('oob: %f' % mean([pq_result_oob[i]['gm'][-1] for i in range(run_num)]))
print('ddm_oci: %f' % mean([pq_result_ddm_oci[i]['gm'][-1] for i in range(run_num)]))
print('hlfr: %f' % mean([pq_result_hlfr[i]['gm'][-1] for i in range(run_num)]))
print('pauc_ph: %f' % mean([pq_result_pauc_ph[i]['gm'][-1] for i in range(run_num)]))
print('dwmil: %f' % mean([pq_result_dwmil[i]['gm'][-1] for i in range(run_num)]))
print('acdwm: %f' % mean([pq_result_acdwm[i]['gm'][-1] for i in range(run_num)]))
