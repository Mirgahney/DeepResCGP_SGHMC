import os
import numpy as np
import pandas as pd
import hydra
import tensorflow as tf

from conv.utils import TqdmExtraFormat as tdqm
from utils import load_data, measure_accuracy, save_result
from models import ResCGPNet
from pdb import set_trace

#TODO:
# 1. try variance update equation and see it's effect
# 2. try different kernels
# 3. go deeper!!
# 4. try challanging datasets ImageNet 32, [corrupted dataset]
# 5. try different train step for sghmc_step and train_hypers *****
# 6. study prediction variance with corrupted dataset
# 7. another architecture design we can add contraction and expansion layers before applying 3x3 conv but need to
#    calculate computation cost to gain from this process *******
# progress bar information print the mll in the progress bar

def init_dataset(cfg):
    (Xtrain, Ytrain), (Xvalid, Yvalid), (Xtest, Ytest) = load_data(cfg.data.name, cfg.data.train_pct, cfg.data.path)
    return (Xtrain, Ytrain), (Xvalid, Yvalid), (Xtest, Ytest)

def init_model(cfg, X, Y):
    #model = hydra.utils.instantiate(cfg.models)
    model = ResCGPNet(X, Y, num_classes=cfg.data.num_classes, layers_strcut=cfg.models.params.layers_strcut,
                      window_size=cfg.models.window_size, expansion_factor=cfg.models.params.expansion_factor,
                      M=cfg.models.params.M, kernel=cfg.models.params.kernel, batch_size=cfg.data.batch_size,
                      lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay,
                      feature_maps=cfg.models.params.feature_maps)
    return model

def creat_task(cfg):
    model_name = f'{cfg.arch}_fm_{cfg.feature_maps}_M{cfg.M}_K{cfg.kernel}_lr{cfg.lr}_bs{cfg.batch_size}_' \
                 f'efactor{cfg.models.expansion_factor}_sghmc{cfg.sghmc_step}'
    save_dir = f'run/{cfg.dataset}/{model_name}'
    writer = tf.compat.v1.summary.FileWriter(f'{save_dir}')
    return save_dir, writer

def train_model(cfg, model, Xvalid, Yvalid, writer, save_dir):
    mll_max = -np.inf
    best_iter = 0
    best_model = model
    for i in tdqm(range(cfg.train.iterations), ascii=" .oO0", bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):
        if i == 0:
            for j in range(cfg.train.sghmc_step):
                model.sghmc_step()
            summary = model.train_hypers(tb=True)
            print("Iteration", i, end='\r')
            mll, sum_mll = model.print_sample_performance(tb=True)
            print(f"MLL: {mll}")
            # set_trace()
            writer.add_summary(summary, global_step=i)
            writer.add_summary(sum_mll, global_step=i)
        else:
            for j in range(cfg.train.sghmc_step):
                model.sghmc_step()
            model.train_hypers()
            print("Iteration", i, end='\r')
            mll, _ = model.print_sample_performance(tb=True)
            print(f"MLL: {mll}")

        if np.round(mll - mll_max, decimals=5) > 0:
            print('MLL increased ({:.7f} --> {:.7f}). Updating values ....'.format(mll_max, mll))
            mll_max = mll
            best_model = model
            best_iter = i

    if cfg.save_model:
        # save best model
        print(f'################## save best model at {save_dir} ##################')
        model_name = f'{best_iter}_bestmodel_{mll_max}'
        best_model.save(save_dir, name=model_name)

    accuracy = measure_accuracy(best_model, Xvalid, Yvalid)
    print(f"Best Model Test accuracy: {accuracy} with MLL {mll_max}")
    # export accuracy
    acc_mll_df = pd.DataFrame([accuracy], columns=['accuracy'])
    acc_mll_df['mll'] = mll_max
    save_result(acc_mll_df, save_dir, name='_mll_accuracy')

@hydra.main(config_path='./config/Residual_learning_config.yaml', strict=False)
def main(cfg):
    # initialize dataset
    (Xtrain, Ytrain), (Xvalid, Yvalid), (Xtest, Ytest) = init_dataset(cfg)

    save_dir, writer = creat_task(cfg)

    # initialize model
    model = init_model(cfg, Xtrain, Ytrain)

    # train model
    train_model(cfg, model, Xvalid, Yvalid, writer, save_dir)

if __name__ == '__main__':
    main()