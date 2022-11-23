# LDRFlow
Implementation of our paper, entitiled Unsupervised Optical Flow Estimation for Differently Exposed Images in LDR Domain, submitted to IEEE TCSVT.

## Requirements
The code has been tested with Python 3.7, PyTorch 1.7, and Cuda 10.2.
```Shell
conda create --name ldrflow
conda activate ldrflow
conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=10.2 -c pytorch
conda install matplotlib
conda install tensorboard
conda install scipy
conda install opencv
```

## Datasets
To train/evaluate, you will need to download the required datasets.
* [SIG17 (Kal17)](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
* [ICCP19 (Pra19)](https://val.cds.iisc.ac.in/HDR/ICCP19/)
* [SIG17 (Kal17)](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
* [SIG17 (Kal17)](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
* [SIG17 (Kal17)](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)

Build the `data` dir, and put the downloaded data in it. By default `datasets.py` will search for the datasets in the location.
```Shell
├── data
    ├── Raw
        ├── Test
        ├── Training
        ├── Training_ICCV
    ├── IMF_short2long
        ├── Test
        ├── Training
        ├── Training_ICCV
    ├── IMF_table
        ├── Training
        ├── Training_ICCV
    ├── GT_flow
```
