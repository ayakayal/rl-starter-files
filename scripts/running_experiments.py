from train_2 import main
import os
import csv
import re
import pandas as pd
import torch
import datetime
import tensorboardX
import matplotlib.pyplot as plt

def run_training(algo, my_env,folder_name,seeds,frames,frames_per_proc, hyperparameter):
    # Loop through each seed and run the script with it
    df_list = []
    for seed in seeds:
        main(algo=algo,env=my_env,folder_name=folder_name,model=f"{my_env}_{algo}_seed{seed}_ir{hyperparameter}",seed=seed,log_interval=1,save_interval=10,procs=1,frames=frames,epochs=4,batch_size=256,frames_per_proc=frames_per_proc,discount=0.99,lr=0.001,gae_lambda=0.95,entropy_coef=0,value_loss_coef=0.5,max_grad_norm=0.5,optim_eps=1e-8,optim_alpha=0.99,clip_eps=0.2,recurrence=1,text=False,ir_coef=hyperparameter)
        file_path = f'storage/{folder_name}/{my_env}_{algo}_seed{seed}_ir{hyperparameter}/log.csv'
        df = pd.read_csv(file_path)
        df['return_mean'] = pd.to_numeric(df['return_mean'], errors='coerce')
        df = df.dropna(subset=['return_mean'])  
        df=df.reset_index(drop=True)
        df_list.append(df['return_mean'])
    concat_df= pd.concat(df_list, axis=1)
    #print('the concatinated dataframe',concat_df)
    mean_reward= concat_df.mean(axis=1)

    fig = plt.figure()
    plt.plot(df['frames'].values, mean_reward.values, '-')
    # set the x and y labels
    plt.xlabel('Frames')
    plt.ylabel('Mean Episodic reward')
    # add a title to the plot
    plt.title('Mean Reward vs Frames')
    # save the figure to a file
   
    directory = f'storage/{folder_name}/{my_env}_{algo}_{hyperparameter}'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    fig.savefig(f'{directory}/return_vs_frames.png')
    concat_df['frames']=df['frames']
    concat_df.to_csv(f'{directory}/return_seeds.csv')
#test the script for empty minigrid
# my_env = 'MiniGrid-Empty-8x8-v0'
# seeds = [1,2,3,4,5]
# frames=100000
# frames_per_proc=32
# #running baseline
# run_training('a2c', my_env, seeds, frames,frames_per_proc,0)

# algos = ['a2c_state_count','a2c_entropy','a2c_icm','a2c_icm_inverse']
# hyperparameters = [0.5,0.1,0.05,0.01,0.005,0.001]
# for algo in algos:
#     for hyperparameter in hyperparameters:
#         run_training(algo, my_env, seeds, frames,frames_per_proc,hyperparameter)

#test the script for keydoor
my_env = 'MiniGrid-DoorKey-8x8-v0'
seeds = [1]
frames=100
frames_per_proc=32
#running baseline
run_training('a2c', my_env,'test_test', seeds, frames,frames_per_proc,0)

# algos = ['a2c_state_count','a2c_entropy','a2c_icm','a2c_icm_inverse']
# hyperparameters = [0.5,0.1,0.05,0.01,0.005,0.001]
# for algo in algos:
#     for hyperparameter in hyperparameters:
#         run_training(algo, my_env, seeds, frames,frames_per_proc,hyperparameter)

