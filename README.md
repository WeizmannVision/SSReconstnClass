# Self-Supervised Image Recnostruction & Classification from fMRI 

***Official Keras implementation for:***
> **Self-Supervised Natural Image Reconstruction and Large-Scale Semantic Classification From Brain Activity** \
> *Guy Gaziv\*, Roman Beliy\*, Niv Granot\*, Assaf Hoogi, Francesca Strappini, Tal Golan, Michal Irani*  

<a href="https://doi.org/10.1016/j.neuroimage.2022.119121" target="_blank"><img src="https://img.shields.io/badge/NeuroImage-paper-b31b1b.svg" height=22.5></a>
<a href="http://www.wisdom.weizmann.ac.il/~vision/SSReconstnClass/" target="_blank"><img src="https://img.shields.io/badge/-Project Page-yellowgreen.svg" height=22.5></a>
<a href="https://video.tau.ac.il/events/index.php?option=com_k2&view=item&id=10112:fmri&Itemid=550" target="_blank"><img src="https://img.shields.io/badge/-Summary Video-yellow.svg" height=22.5></a>

⇨ [**NEW PyTorch implementation**](https://github.com/WeizmannVision/SelfSuperReconst)

## Software Requirements
### OS Requirements
* Linux 7.6.1810

### Python Dependencies
* python 3.6
* tensorflow 1.14
* keras 2.2.4
* pytorch 1.1.0
* numpy 1.16.2
* pandas 0.24.2
* sklearn 0.20.3
* seaborn 0.9.0

## Hardware Requirements
These experiments were conducted on Tesla V100 (16GB). The code can run on any GPU with at least 16GB and CUDA
compatibility >=7.0.

Additionally, enough RAM to support in-memory operations is required (estimated 50G).

## Installation Guide
Run the following commands in **bash** (other shells requires some modification to the following scripts):
```bash
# download the code
git clone https://github.com/WeizmannVision/SSReconstnClass
cd ssReconstnClass
# download the data
wget https://dl.dropboxusercontent.com/s/ttx8q0m8mmaz4id/data.tar.gz
# extract the data
tar -xvf data.tar.gz

# install conda env - assumes conda is already installed
conda create -c pytorch -c defaults -c conda-forge -n bvsr --file env.yml
```
Depending on your internet connection, this may take about 30 minutes.

## Demo
The Demo version will produce results for 'fMRI on ImageNet' dataset (subject 3). To run the demo, run the following commands in **bash** (other shells requires some modification to the following scripts):
```bash
# go to the project directory
cd <PROJECT_DIR>
# activate conda environment
conda activate bvsr
# resolve some CXXABI issues that may arise
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# run code
fMRIonImageNet/Reconstruction/run <RESULTS_DIR> <GPU_ID>
```
Note that `<SOMETHING>` should be replaced in the above example, based on the specific paths and GPU you want to use.

Once done (in our settings it takes ~1.5 hours), the result can be found in the provided `<RESULTS_DIR>`.
The results directory will contain:
* Weights for trained encoder and decoder (in `<RESULTS_DIR>/XX.hdf5`).
* Image reconstructions (in `<RESULTS_DIR>/encdec_stage_1_type_0_repeat_0/test_avg`).
* Classification results:
    - CSV (in `<RESULTS_DIR>/demo_class_acc.csv`).
    - Numpy pickled array (in `<RESULTS_DIR>/demo_class_acc.npz`).
    - Plot (in `<RESULTS_DIR>/classification_results_graph.png`).

## Instructions for use (not in demo mode)
**Note:** this is a demo version and not all options are supported!

1. The full version, will need to have the following under data directory:
    1. `data/train_images` - 1.2M ImageNet training set images (divided into folders by class).
    2. `data/val` - 50k ImageNet validation set images.
    3. `data/Vim1_Files` - Vim-1 dataset (images and fMRI).
2. With the datasets in place, the process for reproducing results is quite similar to the demo. Exceptions:
    1. Run command has more arguments (for ablation studies and choice of subject).
    2. Includes a code for _n_-way identification (Perceptual-Similarity based) that will run under the same command.
3. Code for vim-1 will also be published and will have similar structure and instructions.


##
### Acknowledgments
- This code borrows from [Perceptual Similarity Metric](https://github.com/richzhang/PerceptualSimilarity).
- The original datasets behind the released data derivatives are ([fMRI on ImageNet](https://openneuro.org/datasets/ds001246/versions/1.0.1), and [ILSVRC](https://image-net.org/challenges/LSVRC/index.php)).
##
### Citation
If you find this repository useful, please consider giving a star ⭐️ and citation:
```
@article{Gaziv2022,
	title = {{Self-Supervised Natural Image Reconstruction and Large-Scale Semantic Classification from Brain Activity}},
	author = {Gaziv, Guy and Beliy, Roman and Granot, Niv and Hoogi, Assaf and Strappini, Francesca and Golan, Tal and Irani, Michal},
	journal = {NeuroImage},
	doi = {10.1016/J.NEUROIMAGE.2022.119121},
	year = {2022}
}
```
