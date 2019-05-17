import time
import argparse

import mpi4py.MPI as MPI

import utils
import model_test
import evaluator


def dict_sum(da, db, dt=None):
    dc = {}
    dc.update(da)
    dc.update(db)
    return dc


def test(show_detail):
    sim_nums= [60, 70, 80, 90, 100]
    recm_num = 15
    time_start = time.time()
    for sim_num in sim_nums:
        data_path = 'sample_data.csv'
        rate_list = utils.read_csv(data_path, show_detail)[1:]
        train_data, test_data = utils.split_data(rate_list, 1, show_detail)
        ub_model = model_test.UserBasedModel(train_data, show_detail)

        work_size = len(ub_model.user_list) // comm_size
        if comm_rank == work_size - 1:
            work_range = slice(comm_rank * work_size, None)
        else:
            work_range = slice(comm_rank * work_size, (comm_rank + 1) * work_size)
        ub_model.get_sim(sim_num, work_range)
        recm_table = ub_model.get_recm(recm_num, work_range)
        recm_result = comm.reduce(recm_table, op=op_dict_sum, root=0)

        if comm_rank == 0:
            eva = evaluator.Evaluator(rate_list, test_data, recm_result, recm_num)
            recall = eva.recall()
            precision = eva.precision()
            coverage = eva.coverage()
            popularity = eva.popularity()

            log_list = []
            print('- sim_num: ' + str(sim_num) + ' - recm_num: ' + str(recm_num) + ' -')
            log_list.append('- sim_num: ' + str(sim_num) + ' - recm_num: ' + str(recm_num) + ' -')
            print('Recall:     ' + str(recall))
            log_list.append('Recall:     ' + str(recall))
            print('Precision:  ' + str(precision))
            log_list.append('Precision:  ' + str(precision))
            print('Coverage:   ' + str(coverage))
            log_list.append('Coverage:   ' + str(coverage))
            print('Popularity: ' + str(popularity))
            log_list.append('Popularity: ' + str(popularity))
            utils.write_log('log.txt', log_list)
    time_end = time.time()
    time_used = ("%.2f" % (time_end-time_start))
    print('- time_used: ' + str(time_used))


comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
op_dict_sum = MPI.Op.Create(dict_sum, commute=True)

# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', default='configs/res50.yaml')
# parser.add_argument('-d', '--detail', default='True')
# args = parser.parse_args()
# show_detail = False if args.detail.lower() == 'false' else True

test(False)