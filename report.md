# Report Draft

## Introduction

The goal of Project B is to develop a permutation-invariant Variational Network (VN) for 2D + time cardiac image reconstruction. In real applications time frames can be out of order. 

## Description

I tried out different variational networks with different number of layers to see how much detail a deeper network can catch. To account for the permutation-invariant timeframes a vectorial total variation (VTV) regulariser to still make use of the similarity of the timeframes of each batch. The Fourier transform in the VN leaves the time dimension untouched and the VTV is the only way the time domain is taken into account. 

The program is designed that every necessary parameter can easily be changed in a "Hyperparameter" section, including the regulariser and the loss function.

### Training 

The training was conducted on a 80 / 20 training validation split with two possible loss types.

#### Data Fidelity Loss

The standard data fidelity loss proposed in the project script.

USE_DEEP_SUPERVISION = False

$$
\min_{\Theta} \,\mathbb{E}_{(x,s,M)\sim\mathcal{T}}\;\bigl\lVert\,x_{\Theta}^{K}(s,M)\;-\;x\bigr\rVert_{1}
$$

#### Deep Supervision Loss

The data fidelity loss embedded in a deep supervision manner. [[lecture9.pdf]]

USE_DEEP_SUPERVISION = True

$$
\mathcal{L}_{\text{total}} = \sum_{k=1}^K \exp(-\tau (K - k)) \cdot \left\| \mathbf{x}^{(k)} - \mathbf{x}_{\text{GT}} \right\|_1
$$

### Regularisers

There are three available regularisers.

#### VTV

The VTV regulariser proposed in the project script.

$$
\text{Reg}^k(\mathbf{x}^k)_t = \sum_{i \le n_f} \mathbf{D}^{k,i^T} \left\{ (\mathbf{D}^{k,i}\mathbf{x}_t) \odot \varphi^{k,i} \left( \sqrt{\frac{|\mathbf{D}^{k,i}\mathbf{x}_1|^2 + \cdots + |\mathbf{D}^{k,i}\mathbf{x}_T|^2}{T}} \right) \right\} \quad (25)
$$

#### TV

The isotropic TV regulariser introduced in lecture 5

$$
\|\nabla x\|_1 = \sum_p \sqrt{(\nabla_1x)^2 + (\nabla_2x)^2} 
$$

#### Tikhonov

$$
\text{tikhonov}(\mathbf{x}) = \mathbf{x} * \mathbf{K}
\quad \text{where the kernel is} \quad
\mathbf{K} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}
$$

### Choose Parameters

I tried 3, 5 and 10 layers for the following fixed parameters

Hyperparameters:
Network Parameters

N_LAYERS = 3, 5 or 10
N_FILTERS = 5
FILTER_SZ = 3
REGULARISER = "vtv"

Undersampling and noise parameters

NOISE_STD = 0.05
ACCEL_RATE = 4
CENTER_FRACTION = 0.1
SIGMA = 10

Training parameters

BATCH_SIZE = 4
NUM_EPOCHS = 15
LR = 1e-2
PRINT_EVERY = 10
TRAIN_SPLIT = 0.8
DS_TAU = 0.1
USE_DEEP_SUPERVISION = True
SHOW_VAL_IMAGES = True

![[training_loss_different_n_layers.png | 400]]

Description: Training and Validation loss for fixed parameters with different number of layers

![[nRMSE_loss.png | 400]]

Description: nRMSE for fixed parameters with different number of layers

The loss decreased for deeper networks as expected, when also increasing the number of filters the loss further decreases. However when increasing kernel size the network converged much slower see figure todo, I had to increase the epochs from 15 to 25 to see the convergence however the loss overall was still bigger than for all the other combinations. For most of my computations I settled for a lower number of layers and filters because I trained on my Laptop this limited the computation capacity.

![[training_validation_differnt_n_filter.png | 400]]

Description: Training and Validation loss for fixed parameters with different number of layers compared to the increase of number of filters (from 5 to 10) and the increase of filter size (from 3 x 3 kernel to 4 x 4 kernel) with increased number of epochs

![[mRMSE_different_n_filrters_size.png | 400]]

Description: nRMSE for fixed parameters with different number of layers compared to the increase of number of filters (from 5 to 10) and the increase of filter size (from 3 x 3 kernel to 4 x 4 kernel) with increased number of epochs

![[learning_loss 1.png | 400]]

Description: Training loss for a bigger kernel (4 x 4) with 10 layers, 5 filters and 25 epochs

![[reconstruction loss.png | 400]]

Description: Training loss for a normal kernel (3 x 3) with 10 layers, 5 filters and 25 epochs


## Experimental Setting

The experimental setting consists only of the dataset (2dt_heart.mat) that contains 2D + time ground truth images with shape (128, 128, 11, 375) with the first two being the spatial dimension the second is the time dimension and the fourth are the samples. First I created synthetic data by adding an under-sampling mask and noise. From this training pairs I created a training and validation split with 80% training and 20% validation data. 

### Mask generation

The masks are generated randomly for each sample and handed in to the vn with the sample. To generate these masks I use a fixed center radius and outside of this kernel a Laplace shaped density distribution ([[Laplace-Shaped Density]]) with tuneable parameters. 

$$
p(k_y) = \alpha \exp \left( -\frac{|k_y - n_y/2|}{\sigma} \right)
$$

## Results

### VTV vs TV regulariser

Comparing the VTV regulariser with Total Variation (TV) and Tikhonov regulariser a few notable differences appear. 

Training is much faster for TV and Tikhonov which is expected because they only look at one timeframe at a time in comparison the VTV regulariser checks all frames of the sample to draw a conclusion. We also notice that TV and Tikhonov converge much faster than VTV. TV converges at about 2 epochs and tikhonov has already converged after 1 epoch. In comparison vtv only starts to converge after about 13 epochs with the same hyperparams (except the regulariser of course). 

In all three examples the hyperparameters are the same except for the used regulariser. Used Hyperparameters:
Network Parameters

N_LAYERS = 8
N_FILTERS = 5
FILTER_SZ = 3
REGULARISER = "vtv" or "tv"

Undersampling and noise parameters

NOISE_STD = 0.05
ACCEL_RATE = 4
CENTER_FRACTION = 0.1
SIGMA = 10

Training parameters

BATCH_SIZE = 4
NUM_EPOCHS = 15
LR = 1e-2
PRINT_EVERY = 10
TRAIN_SPLIT = 0.8
DS_TAU = 0.1
USE_DEEP_SUPERVISION = True
SHOW_VAL_IMAGES = True


![[2 - Zettelkasten/1 - Atomic Notes/Assets/learning_loss.png | 400]]

Description: VTV training l1 loss 

![[training_loss 3.png | 400]]

Description: Tikhonov training l1 loss

![[tv_training_loss.png | 400]]

Description: TV training loss

On the other hand the VN with the VTV regulariser has much lower losses than with a TV or Tikhonov regulariser. The VN with the Tikhonov regulariser had a training loss of 0.029170 and a validation loss of 0.029245 and a normalised root mean squared error (nRMSE) of 0.140566. The VN with the TV regulariser had a validation loss of todo and a validation loss of todo with a nRMSE of todo. In comparison the VN with the VTV regulariser had a training loss of 0.019454, a validation loss of 0.019311 and a nRMSE of 0.092765.

Lets take a closer look at the differences between the VN with the TV regulariser and the VN with the VTV regulariser for different noise levels and accelerations. 

We can see that the permutation-invariant VN (with the VTV regulariser) performs much better than the VN that looks at each timeframe independently (with the TV regulariser).

![[training_loss_vtv_vs_tv.png.png | 400]]

Description: Training loss for the VN with the VTV regulariser and the VN with the TV regulariser for different noise and acceleration parameters

![[validation_loss_vtv_vs_tv.png | 400]]

Description: Validation error (nRMSE) for the VN with the VTV regulariser and the VN with the TV regulariser for different noise and acceleration parameters
### Deep Supervision vs no deep supervision

In the end of this project I wanted to see the difference between deep supervision and the normal data fidelity loss. To my surprise the the VN with the normal loss performed better than the VN with deep supervision. With the VN with deep supervision reaching a training loss of 0.019391, validation loss of 0.019653 and a nRMSE of 0.094850. The VN without deep supervision reached a training loss of 0.018163, a validation loss of 0.018055 and a nRMSE of 0.084516. 

The VN with deep supervision converged much faster than the VN without it, it still hasn't reached convergence after 25 epochs. This made deep supervision still very useful for testing the influence of the other parameters, because training took quiet long on my laptop.

Hyperparameters:
Network Parameters

N_LAYERS = 8
N_FILTERS = 5
FILTER_SZ = 3
REGULARISER = "vtv"

Undersampling and noise parameters

NOISE_STD = 0.05
ACCEL_RATE = 4
CENTER_FRACTION = 0.1
SIGMA = 10

Training parameters

BATCH_SIZE = 4
NUM_EPOCHS = 25
LR = 1e-2
PRINT_EVERY = 10
TRAIN_SPLIT = 0.8
DS_TAU = 0.1
USE_DEEP_SUPERVISION = True, False
SHOW_VAL_IMAGES = True

![[training_loss 8.png]]

Description: Training loss of the VN with deep supervision

![[training_loss 9.png]]

Description: Training loss of the VN without using deep supervision.

![[reconstruction_example 10.png]]

Description: Example reconstruction images (without deep supervision)

![[reconstruction_example 9.png]]

Description: Example reconstruction images (without deep supervision)

## Discussion

The VTV that turns the 2D VN to a permutation-invariant VN performs much better than regularisers that don't take the time-dimensions into account. This was expected, on the other hand it was surprising to me that the VN with deep supervision performed worse than the VN without. 

The parameters were chosen in a way that kept the limiting computation power and a reasonable runtime in mind. For future experiments it would be interesting to see how more filters and layers perform and if its still holds for a more complex network that deep supervision performs worse than normal data fidelity loss.

## Bibliography

All information used in this project are available in the course "Model- and Learning-Based Inverse Problems in Imaging"

Generative AI was used for coding assistance.

