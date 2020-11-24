# SG_LC
Code for IROS2020 paper [Semantic Graph Based Place Recognition for 3D Point Clouds](https://ras.papercept.net/proceedings/IROS20/0170.pdf)

![](./doc/pipeline.png)

Pipeline overview.

## Citation
If you think this work is useful for your research, please consider citing:
```
@inproceedings{kong2020semantic,
  title={Semantic Graph based Place Recognition for Point Clouds},
  author={Kong, Xin and Yang, Xuemeng and Zhai, Guangyao and Zhao, Xiangrui and Zeng, Xianfang and Wang, Mengmeng and Liu, Yong and Li, Wanlong and Wen, Feng},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={8216--8223},
  year={2020},
  organization={IEEE}
}
```

## Dataset
Preprocess raw point clouds and generate semantic graph pairs.
#### KITTI

<!--
Todo
#### Ford
1. create SCANS with intensity .mat files by 'create_ijrr_dataset.m'
2. create .bin files with normalized intensity SCANS by 'create_bin_data.m'
3. create test_list, poses and timestamp files by 'gen_sem_ford_graph_pairs' 
-->

### Prepare
1. Generate graphs from semantic point clouds 'yxm'
2. Generate positive and negative graph pairs defined by positive distance threshold. Downsampling negative samples for balance.
'get_pair_list'

### Training
3. 

### Testing
3. Generate graph pairs LIST defined by positive distance threshold and time threshold (filter out easy positive pairs). 'prepare/gen_sem_ford_graph_paris.py'
4. Plot the results for comparision



<!--
Todo
## Other methods
#### Scan Context
##### Ford
1. Generate feature database by 'gen_SC_db_ford.py'
2. Compute distance and plot PR curve by 'eval_SC_list_Ford.py'


#### M2DP
##### Ford
1. Generate feature database by 'evaluate_Ford.m'
2. Compute distance and plot PR curve by 'eval_SC_list_Ford.py'
-->