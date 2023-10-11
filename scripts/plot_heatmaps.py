import csv
import numpy as np
import matplotlib.pyplot as plt
import ast

def process_row(row_number,rows_saved,grid_rows,grid_cols):
    keys=rows_saved[0]
# Convert each string to a tuple
    keys = [ast.literal_eval(item) for item in keys]
    values= rows_saved[row_number]
    values=[float(item) for item in values]
    N = sum(values)
    matrix_dict = np.zeros((grid_rows, grid_cols))
    for key, value in zip(keys, values):
            matrix_dict[key[0],key[1]]= value/N
    # print(matrix_dict[1,10]*N)
    max_matrix_dict= np.max(matrix_dict)
    return matrix_dict,max_matrix_dict
def plot_heatmap(matrix_dict,Vmax,filename):
    fig, ax = plt.subplots()
    # Plot the heatmap
    cax = ax.imshow(matrix_dict, cmap='inferno', interpolation='nearest',vmin=0, vmax=Vmax) # you can choose other color maps like 'coolwarm', 'Blues', etc.
    # Add color bar
    cbar = plt.colorbar(cax, label='Value')
    # plt.title(' Heatmap in Minigrid')
    plt.xlabel('Grid X Coordinate')
    plt.ylabel('Grid Y Coordinate')
    # Turn off the tick labels
    ax.set_xticks([])  # Turn off x-axis tick labels
    ax.set_yticks([])  # Turn off y-axis tick labels
    # Save the plot to a file 
    plt.savefig(filename, format='png', dpi=300)




def read_csv_and_retrieve(file_name,heatmap_file_name,grid_rows,grid_cols):
   

    with open(file_name, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows_saved=[r for r in reader]
        matrix_dict1,max_matrix_dict1=process_row(1,rows_saved,grid_rows,grid_cols)
        matrix_dict2,max_matrix_dict2=process_row(3,rows_saved,grid_rows,grid_cols)
        matrix_dict3,max_matrix_dict3=process_row(5,rows_saved,grid_rows,grid_cols)
        matrix_dict4,max_matrix_dict4=process_row(50,rows_saved,grid_rows,grid_cols)
        matrix_dict5,max_matrix_dict5=process_row(97,rows_saved,grid_rows,grid_cols)
       
        Vmax=max(max_matrix_dict1,max_matrix_dict2,max_matrix_dict3,max_matrix_dict4,max_matrix_dict5)
        plot_heatmap(matrix_dict1,Vmax,f'{heatmap_file_name}_1.png')
        plot_heatmap(matrix_dict2,Vmax,f'{heatmap_file_name}_2.png')
        plot_heatmap(matrix_dict3,Vmax,f'{heatmap_file_name}_3.png')
        plot_heatmap(matrix_dict4,Vmax,f'{heatmap_file_name}_4.png')
        plot_heatmap(matrix_dict5,Vmax,f'{heatmap_file_name}_5.png')

#read_csv_and_retrieve('state_MiniGrid-FourRooms-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv','heatmaps_Deadline/state_visitation_FoorRooms_SC',19,19)     
#read_csv_and_retrieve('ir_MiniGrid-FourRooms-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv','heatmaps_Deadline/intrinsic_rewards_FoorRooms_SC',19,19)     
# read_csv_and_retrieve('state_MiniGrid-FourRooms-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv','heatmaps_Deadline/state_visitation_FoorRooms_DIAYN',19,19)
# read_csv_and_retrieve('ir_MiniGrid-FourRooms-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv','heatmaps_Deadline/intrinsic_rewards_FoorRooms_DIAYN',19,19)     
# read_csv_and_retrieve('state_MiniGrid-FourRooms-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv','heatmaps_Deadline/state_visitation_FoorRooms_ICM',19,19)
# read_csv_and_retrieve('ir_MiniGrid-FourRooms-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv','heatmaps_Deadline/intrinsic_rewards_FoorRooms_ICM',19,19) 
   
#read_csv_and_retrieve('state_MiniGrid-FourRooms-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv','heatmaps_Deadline/state_visitation_FoorRooms_entropy',19,19)
read_csv_and_retrieve('ir_MiniGrid-FourRooms-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv','heatmaps_Deadline/intrinsic_rewards_FoorRooms_entropy',19,19) 