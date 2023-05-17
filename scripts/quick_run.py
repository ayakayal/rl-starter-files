#this file is a quick test of train2 
from train_2 import main
#test_speed_no_parallel
# algorithm='a2c'
# env='MiniGrid-DoorKey-8x8-v0'
# folder_name='test_speed_noparallel'
# model=None
# seed=1
# log_interval=1
# save_interval=10
# procs=1
# frames=1000000
# epochs=4
# batch_size=256
# frames_per_proc=32
# discount=0.99
# lr=0.001
# gae_lambda=0.95
# entropy_coef=0
# value_loss_coef=0.5
# max_grad_norm=0.5
# optim_eps=1e-8
# optim_alpha=0.99
# clip_eps=0.2
# recurrence=1
# text=False
# ir_coef=0.001

#testspeedparallel
# algorithm='a2c'
# env='MiniGrid-DoorKey-8x8-v0'
# folder_name='test_speed_parallel'
# model='keydoor_20M'
# seed=1
# log_interval=1
# save_interval=10
# procs=16
# frames=20000000 #it was 1M
# epochs=4
# batch_size=256
# frames_per_proc=32
# discount=0.99
# lr=0.001
# gae_lambda=0.95
# entropy_coef=0
# value_loss_coef=0.5
# max_grad_norm=0.5
# optim_eps=1e-8
# optim_alpha=0.99
# clip_eps=0.2
# recurrence=1
# text=False
# ir_coef=0.001

algorithm='a2c'
env='MiniGrid-DoorKey-8x8-v0'
folder_name='aya'
model=None
seed=1
log_interval=1
save_interval=10
procs=1
frames=20000 #it was 1M
epochs=4
batch_size=256
frames_per_proc=32
discount=0.99
lr=0.001
gae_lambda=0.95
entropy_coef=0
value_loss_coef=0.5
max_grad_norm=0.5
optim_eps=1e-8
optim_alpha=0.99
clip_eps=0.2
recurrence=1
text=False
ir_coef=0.001


main(algorithm=algorithm,env=env,folder_name=folder_name,model=model,seed=seed,log_interval=log_interval,save_interval=save_interval,procs=procs,frames=frames,epochs=epochs,batch_size=batch_size,frames_per_proc=frames_per_proc,discount=discount,lr=lr,gae_lambda=gae_lambda,entropy_coef=entropy_coef,value_loss_coef=value_loss_coef,max_grad_norm=max_grad_norm,optim_eps=optim_eps,optim_alpha=optim_alpha,clip_eps=clip_eps,recurrence=recurrence,text=text,ir_coef=ir_coef)
