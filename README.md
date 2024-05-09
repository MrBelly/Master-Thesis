# Master's Thesis: The Lightweight Monocular Depth Estimation

**Author:** Mehmet Omer Eyi

 ## Problem Description and Objective

The main challenge in designing a monocular depth estimation is providing a low--
complex model while maintaining its performance in generating depth maps. Intro-
reducing Multi Head Self Attention (MHSA) and vision transformer modules to the
depth model is not desired due to their high complexities. Instead of using these
modules, Zhang et al. use cross-covariance attention [14] to alleviate the computation
burden in attention layers, whereas Ning et al. use trap attention techniques to tackle
this complexity issue [17]. Apart from these recent attempts to solve the complexity problem, 
novel ways to solve this problem will be investigated in this thesis. In
this thesis, Zhang et al.’s lightweight monocular depth estimation network is selected
as the baseline model while comparing the performance and complexity of the pro-
posed models. This thesis proposes novel approaches in the encoder and decoder of
the depth model. In the encoder of the depth model, Zhang et al.’s attention mod-
rules are replaced with lighter-attention and MLP-based modules. The complexity and
performance of the modified depth estimation networks are evaluated and compared
with this baseline. Apart from these modifications in the depth model’s encoder, an
up-sampling technique having a low complex attention module is designed for the decoder 
of the depth model, and the bi-linear interpolation operations in the decoder
are replaced with this attention-based up-sampling technique. Pixel shuffle operation
is employed to alleviate the complexity of the attention module introduced by this
up-sampling technique. This up-sampling technique’s impacts on model complexity
and performance are investigated.

For KITTI dataset preparation, please refer to the [Monodepth repository](https://github.com/nianticlabs/monodepth2).

The [Make3D dataset](http://make3d.cs.cornell.edu/data.html) can be accessed here.

Checkpoints are available in the [chkps](https://drive.google.com/drive/folders/1-smHjqesz2kR1EoDAMLq0UPjgcgGOlKr?usp=drive_link).

The presentation and report are available in the [documents](https://drive.google.com/drive/folders/1GAAg6Xuo40oYEsniGuuuYDjmV5iPLaB3?usp=sharing)

The Python dependencies for the training and the testing of the models are in the depth_anaconda_env.
