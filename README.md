# SDBS-Transformer (Sparse Deformable Transformer with Background Suppression for Crowd Localization)

An official implementation of "Sparse Deformable Transformer with Background Suppression for Crowd Localization" . 

# Environment

We have tested the code on the following environment: 

Python 3.8.3 / Pytorch 1.8.1 / torchvisoin 0.9.1 / CUDA 11.1 / Ubuntu 18.04

## Run the following command to install dependencies:

```
pip install -r requirements.txt
```

## Compiling CUDA operators

```
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

# Datasets
- Download JHU-CROWD ++ dataset from [here](http://www.crowd-counting.com/)  

- Download UCF-QNRF dataset from [here]( https://www.crcv.ucf.edu/data/ucf-qnrf/)

- Download NWPU-Crowd dataset  from [here](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)  

  

# Prepare data
## Generate point map
```cd SDBS/data```  
For JHU-Crowd++ dataset: ```python prepare_jhu.py --data_path /xxx/xxx/jhu_crowd_v2.0```  
For NWPU-Crowd dataset: ```python prepare_nwpu.py --data_path /xxx/xxx/NWPU_SDBS```

For UCF-QNRF dataset: ```python prepare_qnrf.py --data_path /xxx/xxx/UCF-QNRF_ECCV18```  
For ShanghaiTech dataset: ```python prepare_sh.py --data_path /xxx/xxx/ShanghaiTech```

## Generate image list
```cd SDBS ```   
```python make_npydata.py --jhu_path /xxx/xxx/jhu_crowd_v2.0 --nwpu_path /xxx/xxx/NWPU_SDBS --qnrf_path /xxx/xxx/UCF-QNRF_ECCV18 --SH_path /xxx/xxxShanghaiTech```

# Training 
Example:  
```cd SDBS```  
```sh config/jhu/r50_SDBS_rho_0.1.sh```   

or

```sh config/nwpu/r50_SDBS_rho_0.1.sh```   

* The model will be saved in ```SDBS/res/xxx.pth```  
* The log will be saved in ```SDBS/log_file/debug.txt```  

# Testing
Example:  
```python test.py --dataset jhu --pre model.pth --gpu_id 0```   
or  
```python test.py --dataset nwpu --pre model.pth --gpu_id 0``` 

* The model.pth can be obtained from the training phase.



## Visual

Example:  
```python vis_photo.py  --pre model.pth --photo xxx.jpg --gpu_id 0```   

# Acknowledgement
Thanks for the following great work:

```
@inproceedings{roh2022sparse,
  title={Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity},
  author={Roh, Byungseok and Shin, JaeWoong and Shin, Wuhyun and Kim, Saehoon},
  booktitle={ICLR},
  year={2022}
}
```

```
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```

```
@article{liang2022end,
  title={An end-to-end transformer model for crowd localization},
  author={Liang, Dingkang and Xu, Wei and Bai, Xiang},
  journal={European Conference on Computer Vision},
  year={2022}
}
```
# Reference
If you find this project is useful, please cite:
```
@article{Ge2024end,
  title={Sparse Deformable Transformer with Background Suppression for Crowd Localization},
  author={},
  journal={},
  year={2024}
}
```
