# LDRFlow
Implementation of our paper, entitiled Unsupervised Optical Flow Estimation for Differently Exposed Images in LDR Domain, submitted to IEEE TCSVT.
## 1. Requirements
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
## 2. Datasets
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

## 3. Preprocessing
### 3.1 IMF-based brightness normalization
It is better to perform the IMF-based brightness normalization in advance, since the code is based on Matlab. You can the run a demo of IMF by `data/IMF_demo.m`. We also implement the IMF using PyTorch, such that the brightness normalization can run on-the-fly. The demo of IMF in PyTorch can be found at `core/imf_utils.py`

To generate the normalized images and the histogram tables of IMF, run `data/preprocess.m` for the Test, Training, and Training_ICCV data. Then, the generated images and tables are stored into `data/IMF_short2long` and `data/IMF_table`, respectively.

Notably, for two images ''filename1.tif'', ''filename2.tif'', if the first image is normalized to the second image, then the normalized image is saved as ''filename1-filename2.tif''. The corresponding IMF table is saved as ''filename1-filename2.mat''.
### 3.2 Ground truth
We use two traditional methods to obtain the ground truth optical flows of test data:
* [FullFlow](https://cqf.io/fullflow/)
* [MirrorFlow](https://bitbucket.org/visinf/projects-2017-mirrorflow/src/master/)

## 4. Training
When the data is prepaired, run the following command:
```Shell
python train.py --name ablation6 --stage ablation --num_steps 100000 --batch_size 3 --lr 0.00025 --image_size 384 704 --wdecay 0.0001
python train.py --name ldrflow-sota --stage sota --num_steps 150000 --batch_size 12 --lr 0.0004 --image_size 384 704 --wdecay 0.0001
```
Training logs will be written to the `runs` which can be visualized using tensorboard.

## 5. Test
For test, run the following command:
```Shell
python evaluate.py --model=checkpoints/ldrflow-sota.pth --mixed_precision
```
Our pretraind model is provided [here](https://drive.google.com/drive/folders/1XmpI3ldHEaqXdE-6vCxLzSTIb7Z8F5yn?usp=sharing).

