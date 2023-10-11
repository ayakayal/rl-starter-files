
import os
import subprocess
import shutil

# Define the values of ir-coef that you want to test
ir_coefs = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005]

# Define the directory to save the models in
save_dir = "./storage/save_models_Doorkey_state"

# Create the save directory if it doesn't already exist
os.makedirs(save_dir, exist_ok=True)

# Loop over the ir-coef values and run the command with each value
for i, ir_coef in enumerate(ir_coefs):
    # Define the name of the model for this run
    model_name = f"sim_SC_{ir_coef}"
    
    # Define the command to run
    command = f"python3 -m scripts.train --algo a2c_state_count --env MiniGrid-DoorKey-8x8-v0 --model {model_name} --save-interval 10 --frames 100000 --entropy-coef=0 --procs=1 --frames-per-proc=32 --ir-coef={ir_coef}"
    
    # Run the commands

    subprocess.run(command.split())
    
    # Move the saved model to the save directory with a unique name
   
    model_folder = f"{model_name}"

    new_model_folder = os.path.join(save_dir, f"{model_folder}")

    shutil.move(os.path.join('/home/rmapkay/rl-starter-files/storage', f"{model_folder}"), new_model_folder)