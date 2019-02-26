# MisGAN: Learning from Incomplete Data with GANs

This repository provides a PyTorch implementation of
[MisGAN](https://openreview.net/forum?id=S1lDV3RcKm&noteId=S1lDV3RcKm),
a GAN-based framework for learning from incomplete data.


## Requirements

The code requires Python 3.6 or later.
The file [requirements.txt](requirements.txt) contains the full list of
required Python modules.


## Jupyter notebook

We provide a [notebook](misgan.ipynb) that includes an overview of MisGAN
as well as the annotated implementation that runs on MNIST.
The notebook can be viewed from
[here](https://nbviewer.jupyter.org/github/steveli/misgan/blob/master/misgan.ipynb).


## Usage

The source code are under the `src` directory.
Separate scripts are provided to run MisGAN on MNIST and CelebA datasets.

For CelebA, you will need to download the dataset from its
[website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):

* Download the file `img_align_celeba.zip` (available from [this link](https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM)).
* Unzip and place the folder under the `src/celeba-data` directory.

The commands below need to be run under the `src` directory.

MisGAN on MNIST:
```bash
python mnist_misgan.py
```

MisGAN imputation on MNIST:
```bash
python mnist_misgan_impute.py
```

MisGAN on CelebA:
```bash
python celeba_misgan.py
```

MisGAN imputation on CelebA:
```bash
python celeba_misgan_impute.py
```

See more options with the command line argument `-h` for each script.


## References

Steven Cheng-Xian Li, Bo Jiang, Benjamin Marlin.
"MisGAN: Learning from Incomplete Data with Generative Adversarial Networks."
ICLR 2019.
\[[OpenReview](https://openreview.net/forum?id=S1lDV3RcKm&noteId=S1lDV3RcKm)\]


## Contact

Your feedback would be greatly appreciated!
Reach us at cxl@cs.umass.edu
