# SG_PR

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

## Requirements
We recommend python3.6. You can install required dependencies by:
```bash
pip install -r requirements.txt
```

## Training

### Data structure

The data structure is:

```bash
data
    |---00
    |    |---000000.json   
    |    |---000001.json
    |    |---...
    |
    |---01
    |    |---000000.json
    |    |---000001.json
    |    |---...
    |
    |---...
    |
    |---00.txt
    |---01.txt
    |---...
```
You can download the provided [preprocessed data](https://drive.google.com/file/d/1eu4G008gvAJGjU-M8qBvTN0JLHG2B-OB/view?usp=sharing).
Or you can refer to the 'data_process' dir for details of generating graphs.

### Configuration file

Before training the model, you need to modify the configuration file in ./config according to your needs. The main parameters are as follows:
- graph_pairs_dir: the root dir of your dataset.
- batch_size: batch size, 128 in our paper.
- p_thresh: distance threshold for positive samples, e.g., 3m 5m 10m. If the distance between two samples is less than p_thresh meters, they will be treated as positive samples. The distance threshold for negative samples is set to 20 meters by default. Note that your training sample pairs should not contain samples with a distance greater than p_thresh meters and less than 20 meters.

### Training model

- model: pre-trained model.
- train_sequences: list of training data.
- eval_sequences: list of validation data.
- logdir: path to save training results.
- graph_pairs_dir: set to SK label dir, e.g., '../SG_PR_DATA/graphs_sk'
- pair_list_dir: set to train dir, e.g., '../SG_PR_DATA/train/3_20', 3_20 refers to positive threshold 3m negative threshold 20m

After preparing the data and modifying the configuration file, you can start training. Just run:

```bash
python main_sg.py
```

## Testing

### eval_pair

This example takes a pair of graphs as input and output their similarity score. To run this example, you need to set the following parameters:
- model: the eval model file.
- pair_file: a pair of graph.

Then just run:
```bash
python eval_pair.py
```

### eval_batch

This example tests a sequence, the results are it's PR curve and F1 max score. To run this example, you need to set the following parameters:
- model: the eval model file.
- graph_pairs_dir: set to SK label dir ('../SG_PR_DATA/graphs_sk') or RN prediction dir ('../SG_PR_DATA/graphs_rn') or other semantic prediction in the future.
- pair_list_dir: set to eval dir which excludes easy positive pairs, e.g., '../SG_PR_DATA/eval/3_20' 
- sequences: list of test sequences.
- output_path: path to save test results.
- show: whether to display the pr curve in real time.

Then just run:
```bash
python eval_batch.py
```

### Raw Data
We provide the [raw data](https://drive.google.com/file/d/1R6d13HOtR6y2wrrXAaeXOrmJXeJfzyoq/view?usp=sharing) of the tables and curves in the paper, including compared methods M2DP, PointNetVLAD, Scan Context.

We recommend users refer the work [SSC](https://github.com/lilin-hitcrt/SSC#raw-data) for a fair comparison with recent methods in the same data distribution.

## Other methods
#### PointNetVLAD
Please refer to our modified repo for training and testing PointNetVLAD on KITTI dataset, which is mentioned in our paper
as [PV_KITTI](https://github.com/kxhit/pointnetvlad).

<!--
Todo
#### Scan Context
##### Ford
1. Generate feature database by 'gen_SC_db_ford.py'
2. Compute distance and plot PR curve by 'eval_SC_list_Ford.py'


#### M2DP
##### Ford
1. Generate feature database by 'evaluate_Ford.m'
2. Compute distance and plot PR curve by 'eval_SC_list_Ford.py'
-->

## TODO
- [ ] Support Ford Campus Dataset
- [x] Release compared methods e.g., [PointNetVLAD](https://github.com/mikacuy/pointnetvlad) trained on KITTI (PV-KITTI)
- [x] Release preprocessing code


## Acknowledgement

Thanks to the source code of some great works such as [SIMGNN](https://github.com/benedekrozemberczki/SimGNN), [DGCNN](https://github.com/WangYueFt/dgcnn).