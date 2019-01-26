# QuaternionCNN_Keras
Implementation of Quaternion Convolutional Neural Networks 

## Requirements
Keras(The writer's version is 2.1.3, with Tensorflow backend 1.4.1)

Numpy

Scipy

PIL

Scikit-image
## quaternion_layers 
It contains implementation of quaternion version of convolutional layer and fully-connected layer, called QConv and QDense.
They can be applied in same way as normal layers, note that they will consider the inputs and outputs as pure quaternions, which means the number of input/output channels must be multiple of three.

This code is heavily borrowed from [Deep Complex Networks](https://github.com/ChihebTrabelsi/deep_complex_networks)
## cifar10_cnn.py
It's the one of the examples that Keras gives. The layers are replaced by quaternion layers.
## denoising.py
It uses a U-net for denoising. A training set and a validation set are needed to run this file. The input size is 128x128. 

The training images should be in a folder named 'dataset' and must be at least 128x128. 128x128 patches will be randomly cropped when training. 

The validation images should be in a folder named 'validation_split' and must be 128x128.
## Citation
Please cite our work as
```
@inproceedings{zhu2018quaternion,
  title={Quaternion Convolutional Neural Networks},
  author={Xuanyu Zhu, Yi Xu, Hongteng Xu and Changjian Chen},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```
