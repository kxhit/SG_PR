# SG_LC
Semantic Graph Based Place Recognition for 3D Point Clouds

### dataset
#### KITTI
#### Ford
1. create SCANS with intensity .mat files by 'create_ijrr_dataset.m'
2. create .bin files with normalized intensity SCANS by 'create_bin_data.m'
3. create test_list, poses and timestamp files by 'gen_sem_ford_graph_pairs' 

### Prepare
1. Generate graphs from semantic point clouds 'yxm'
2. Generate positive and negative graph pairs defined by positive distance threshold. Downsampling negative samples for balance.
'get_pair_list'
### Training
3. 

### Eval
3. Generate graph pairs LIST defined by positive distance threshold and time threshold (filter out easy positive pairs). 'prepare/gen_sem_ford_graph_paris.py'
4. Plot the results for comparision

### Other methods
#### Scan Context
##### Ford
1. Generate feature database by 'gen_SC_db_ford.py'
2. Compute distance and plot PR curve by 'eval_SC_list_Ford.py'


#### M2DP
##### Ford
1. Generate feature database by 'evaluate_Ford.m'
2. Compute distance and plot PR curve by 'eval_SC_list_Ford.py'