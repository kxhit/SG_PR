# SGPR

## Description
This is the source code of "Semantic Graph Based Place Recognition for 3D Point Clouds"

## Environment
- Ubuntu 16.04(Recommended)
- CUDA 10.0/10.1
- Python 2 (running as ROS node) / Python 3 (training)

## Requirements

- ros-kinetic
- pytorch==1.1
- torchvision==0.3.0.0
- pypcl
- numpy
- pykitti
- tqdm

 and more

## Data Preparation

1. Generate semantic graphs.
```bash
$ source /opt/ros/kinetic/setup.bash
$ python gen_label_graph.py -d $DATASET_DIR
```
It will assign semantic label and instance label for each point, and organize the labeled points into graphs. Users can choose to output to a folder or publish on ros topics, or both of them. With `--label_topic` and `--graph_topic` argument, users can change the publish topic. `--demolition` argument is for 'Robustness Test' in our paper.

2. Construct semantic graph pairs.

To generate semantic graph pairs from scratch, run:

```bash
$ python gen_sem_kitti_graph_pairs.py -d $DATASET_DIR -g $GRAPH_DIR -o $OUTPUT_DIR
```
Or we provide pair lists prepared in advance to generate semantic graph pairs faster.

```bash
$ python gen_sem_kitti_graph_pairs_fast.py -d $DATASET_DIR -g $GRAPH_DIR -o $OUTPUT_DIR
```
##  TODO
1. Ring-based clustering.
2. Some visualization code.
3. Test the code.