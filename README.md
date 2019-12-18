# STGAN (CVPR 2019)

An unofficial **PyTorch**  implementation of [**STGAN: A Unified Selective Transfer Network for Arbitrary Image Attribute Editing**](https://arxiv.org/abs/1904.09709). 

## Requirements
- [Python 3.6+](https://www.python.org)
- [PyTorch 1.0+](https://pytorch.org)
- [tensorboardX 1.6+](https://github.com/lanpa/tensorboardX)
- [torchsummary](https://github.com/sksq96/pytorch-summary)
- [tqdm](https://github.com/tqdm/tqdm)
- [Pillow](https://github.com/python-pillow/Pillow)
- [easydict](https://github.com/makinacorpus/easydict)

## Sample

From left to right: Origin, Bangs, Blond_Hair,  Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, Pale_Skin, Young.

![](sample.jpg)

## Preparation

Please download the [CelebA](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf) dataset from this [project page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Then organize the directory as:

```
├── data_root
│   └── image
│       ├── 000001.jpg
│       ├── 000002.jpg
│       ├── 000003.jpg
│       └── ...
│   └── anno
│       ├── list_attr_celeba.txt
│       └── ...
```

## Training

- For quickly start, you can simply use the following command to train:

  ```console
  CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config ./configs/train_stgan.yaml
  ```

- If you want to modify some hyper-parameters, please edit them in the configuration file `./configs/train_stgan.yaml` following the explanations below:
  - `exp_name`: the name of current experiment.
  - `mode`: 'train' or 'test'.
  - `cuda`: use CUDA or not.
  - `ngpu`: how many gpu cards to use. Notice: this number should be no more than the length of CUDA_VISIBLE_DEVICES list.
  - `dataset`: the name of dataset. Notice: you can extend other datasets.
  - `data_root`: the root of dataset.
  - `crop_size`: the crop size of images.
  - `image_size`: the size of input images during training.
  - `g_conv_dim`: the base filter numbers of convolutional layers in G.
  - `d_conv_dim`: the base filter numbers of convolutional layers in D.
  - `d_fc_dim`: the dimmension of fully-connected layers in D.
  - `g_layers`: the number of convolutional layers in G. Notice: same for both encoder and decoder.
  - `d_layers`: the number of convolutional layers in D.
  - `shortcut_layers`: the number of shortcut connections in G. Notice: also the number of STUs.
  - `stu_kernel_size`: the kernel size of convolutional layers in STU.
  - `use_stu`: if set to false, there will be no STU in shortcut connections.
  - `one_more_conv`: if set to true, there will be another convolutional layer between the decoder and generated image.
  - `attrs`: the list of all selected atrributes. Notice: please refer to `list_attr_celeba.txt` for all avaliable attributes.
  - `checkpoint`: the iteration step number of the checkpoint to be resumed. Notice: please set this to `~` if it's first time to train.
  - `batch_size`: batch size of data loader.
  - `beta1`: beta1 value of Adam optimizer.
  - `beta2`: beta2 value of Adam optimizer.
  - `g_lr`: the base learning rate of G.
  - `d_lr`: the base learning rate of D.
  - `n_critic`: number of D updates per each G update.
  - `thres_int`: the threshold of target vector during training.
  - `lambda_gp`: tradeoff coefficient of D_loss_gp.
  - `lambda1`: tradeoff coefficient of D_loss_att.
  - `lambda2`: tradeoff coefficient of G_loss_att.
  - `lambda3`: tradeoff coefficient of G_loss_rec.
  - `max_iters`: maximum iteration steps.
  - `lr_decay_iters`: iteration steps per learning rate decay.
  - `summary_step`: iteration steps per summary operation with tensorboardX.
  - `sample_step`: iteration steps per sampling operation.
  - `checkpoint_step`: iteration steps per checkpoint saving operation.

## Acknowledgements

This code refers to the following two projects:

[1] [TensorFlow implementation of STGAN](https://github.com/csmliu/STGAN) 

[2] [PyTorch implementation of StarGAN](https://github.com/yunjey/stargan)
