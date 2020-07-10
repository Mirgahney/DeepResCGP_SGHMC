import os
import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf

from conv.utils import TqdmExtraFormat as tdqm
from utils import load_data, measure_accuracy, save_result
from models import ResCGPNet8, ResCGPNet11, ResCGPNet17, PlainCGPNet11
import argparse
from pdb import set_trace

#TODO:
# 1. try variance update equation and see it's effect
# 2. try different kernels
# 3. go deeper!!
# 4. try challanging datasets ImageNet 32, [corrupted dataset]
# 5. try different train step for sghmc_step and train_hypers *
# 6. study prediction variance with corrupted dataset
# 7. another architecture design we can add contraction and expansion layers before applying 3x3 conv but need to
#    calculate computation cost to gain from this process **
# 8. experiment with l.Z alone and with l.U and l.Z make l.U trainable
# progress bar information print the mll in the progress bar

def train_model(cfg, model, Xvalid, Yvalid, writer, save_dir):
    mll_max = -np.inf
    best_iter = 0
    accuracy_list = []
    mll_list_que = deque([mll_max, mll_max, mll_max])
    best_model_que = deque([model, model, model])
    for i in tdqm(range(cfg.iterations), ascii=" .oO0", bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):
        if i == 0:
            for j in range(cfg.sghmc_step):
                model.sghmc_step()
            summary = model.train_hypers(tb=True)
            print("Iteration", i, end='\r')
            mll, sum_mll = model.print_sample_performance(tb=True)
            print(f"MLL: {mll}")
            # mll = struct.unpack('fff', sum_mll)[2]
            # set_trace()
            writer.add_summary(summary, global_step=i)
            writer.add_summary(sum_mll, global_step=i)
        else:
            for j in range(cfg.sghmc_step):
                model.sghmc_step()
            model.train_hypers()
            print("Iteration", i, end='\r')
            mll, _ = model.print_sample_performance(tb=True)
            print(f"MLL: {mll}")

        if np.round(mll - mll_max, decimals=5) > 0:
            # accuracy = measure_accuracy(model)
            mll_max = mll

            print('MLL increased ({:.7f} --> {:.7f}). Updating values ....'.format(mll_list_que[-1], mll_max))
            mll_list_que.append(mll)
            best_model_que.append(model)  # append best model so far
            best_model_que.popleft()  # remove worst model so far
            mll_list_que.popleft()
            best_iter = i

    # save best model
    print(f'################## save best model at {save_dir} ##################')
    model_name = f'{best_iter}_bestmodel_{mll_list_que[-1]}'
    best_model_que[-1].save(save_dir, name=model_name)

    accuracy = measure_accuracy(best_model_que[-1], Xvalid, Yvalid)
    accuracy_list.append(accuracy)
    print("Best Model Test accuracy:", accuracy)

    acc_mll_df = pd.DataFrame(accuracy_list, columns=['accuracy'])
    acc_mll_df['mll'] = mll_list_que[-1]
    save_result(acc_mll_df, save_dir, name='_mll_accuracy')

parser = argparse.ArgumentParser()
parser.add_argument('--feature_maps', default=10, type=int)
parser.add_argument('-M', default=64, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--iterations', default=100000, type=int)
parser.add_argument('--dataset', default = "mnist", choices=['mnist', 'fashion_mnist', 'cifar'], type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--load', action='store_true')
parser.add_argument('--kernel', default='rbf', choices=['rbf', 'matern12', 'matern32', 'matern52'], type=str)
parser.add_argument('--arch', default='ResCGPNet8', choices=['ResCGPNet8', 'ResCGPNet11', 'ResCGPNet17', 'PlainCGPNet11'], type=str)
parser.add_argument('--train-pct', default=1.0, type=float)
parser.add_argument('--sghmc-step', default=1, type=int, help='number of sghmc update steps')

cfg = parser.parse_args()

(Xtrain, Ytrain), (Xvalid, Yvalid), (Xtest, Ytest) = load_data(cfg.dataset, cfg.train_pct)

if cfg.arch == 'ResCGPNet8':
    model = ResCGPNet8(cfg, Xtrain, Ytrain, num_classes=10, window_size=100, expansion_factor=0)
elif cfg.arch == 'ResCGPNet11':
    model = ResCGPNet11(cfg, Xtrain, Ytrain, num_classes=10, window_size=100, expansion_factor=0)
elif cfg.arch == 'ResCGPNet17':
    model = ResCGPNet17(cfg, Xtrain, Ytrain, num_classes=10, window_size=100, expansion_factor=0)
elif cfg.arch == 'PlainCGPNet11':
    model = PlainCGPNet11(cfg, Xtrain, Ytrain, num_classes=10, window_size=100)#, expansion_factor=0)
else:
    raise Exception('Undefined network architecture')

save_dir = f'run/{cfg.dataset}/{cfg.arch}_fm_{cfg.feature_maps}_M{cfg.M}_K{cfg.kernel}_lr{cfg.lr}_bs{cfg.batch_size}' \
           f'_sghmc{cfg.sghmc_step}'

if cfg.load:
    print("Loading parameters")
    #set_trace()
    checkpoint = tf.train.latest_checkpoint(save_dir)
    model._saver.restore(model.session, checkpoint)
    model.sghmc_step()
    model.train_hypers()
    accuracy = measure_accuracy(model, Xtest, Ytest)
    print(f"Model {save_dir} Test accuracy:", accuracy)

# create a data frame to save intermediate resulr
result_df = pd.DataFrame(columns=['step', 'mll'])#, 'accuracy'])

writer = tf.compat.v1.summary.FileWriter(f'{save_dir}')#, tf.get_default_graph())

train_model(cfg, model, Xvalid, Yvalid, writer, save_dir)