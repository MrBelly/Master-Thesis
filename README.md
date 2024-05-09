# Master's Thesis: The Lightweight Monocular Depth Estimation

**Author:** Mehmet Omer Eyi

 ## Problem Description and Objective

The main challenge in designing a monocular depth estimation is providing a low-complex
model while maintaining its performance in generating depth maps. Introducing 
Multi Head Self Attention (MHSA) and vision transformer modules to the
depth model is not desired due to their high complexities. Instead of using these
modules, Zhang et al. use cross-covariance attention to alleviate the computation
burden in attention layers, whereas Ning et al. use trap attention techniques to tackle
this complexity issue. Apart from these recent attempts to solve the complexity problem, 
novel ways to solve this problem will be investigated in this thesis. In
this thesis, Zhang et al.’s lightweight monocular depth estimation network is selected
as the baseline model while comparing the performance and complexity of the 
proposed models. This thesis proposes novel approaches in the encoder and decoder of
the depth model. In the encoder of the depth model, Zhang et al.’s attention modules are 
replaced with lighter-attention and MLP-based modules. The complexity and
performance of the modified depth estimation networks are evaluated and compared
with this baseline. Apart from these modifications in the depth model’s encoder, an
up-sampling technique having a low complex attention module is designed for the decoder 
of the depth model, and the bi-linear interpolation operations in the decoder
are replaced with this attention-based up-sampling technique. Pixel shuffle operation
is employed to alleviate the complexity of the attention module introduced by this
up-sampling technique. This up-sampling technique’s impacts on model complexity
and performance are investigated.

## Dataset Preparation

KITTI and Make3d datasets are used in this thesis. The KITTI dataset is used in the training and evaluation whereas Make3D is used only in the evaluation of the models to demonstrate the models' capabilities in generating generalized depth maps  

For KITTI dataset preparation, please refer to the [Monodepth repository](https://github.com/nianticlabs/monodepth2).
The [Make3D dataset](http://make3d.cs.cornell.edu/data.html) can be accessed here.

## Checkpoints
Checkpoints are available in the [chkps](https://drive.google.com/drive/folders/1-smHjqesz2kR1EoDAMLq0UPjgcgGOlKr?usp=drive_link).

## Documents
The presentation and report are available in the [documents](https://drive.google.com/drive/folders/1GAAg6Xuo40oYEsniGuuuYDjmV5iPLaB3?usp=sharing)

## Installation
The Python dependencies for the training and the testing of the models are in the depth_anaconda_env.

## Commands

Pre-training on ImageNet 

```bash
python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --data_path /home/woody/iwnt/iwnt106h/Imagenet/ \
    --epochs 150 \
    --auto_resume True

Training 

python train.py --data_path /home/woody/iwnt/iwnt106h/kitti_data/ --model_name my_train --load_weights_folder /home/hpc/iwnt/iwnt106h/Swift_Former_Lite_Mono/latest_weights/ --split eigen_zhou --num_epochs 60 --batch_size 12  --lr 0.0001 5e-6 31 0.0001 1e-5 31

Evaluation

python evaluate_depth.py --data_path /home/woody/iwnt/iwnt106h/kitti_data/ --load_weights_folder /home/vault/iwnt/iwnt106h/Modified_Mixer_chkp/MLP_Wave/Run_2/my_train/models/weights_45/

