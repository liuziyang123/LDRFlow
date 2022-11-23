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
* [Sen12](https://web.ece.ucsb.edu/~psen/hdrvideo)
* [Tursun16](https://user.ceng.metu.edu.tr/~akyuz/files/eg2016/index.html)
* [ICCV21](https://guanyingc.github.io/DeepHDRVideo/) 

`cd data`, and build the directories as follows:
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

Once the raw data is downloaded, do the following:

* put the training split of SIG17 and the training split of ICCP19 into `data/Raw/Training`.
* put the ICCV21 into `data/Raw/Training_ICCV`.
* put the test splits of SIG17 (PAPER and EXTRA), the test split of ICCP19, the Sen12, and the Tursun16 into `data/Raw/Test`
