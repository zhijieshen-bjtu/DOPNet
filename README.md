Pytorch implementation of ["Disentangling Orthogonal Planes for Indoor Panoramic Room Layout Estimation with Cross-Scale Distortion Awareness"](https://arxiv.org/abs/2303.00971) (CVPR'23)  
The project is reproducted based on LGT-Net and [PanoFormer](https://github.com/zhijieshen-bjtu/PanoFormer). 
# Trained Models 
Download the model at this [link](https://drive.google.com/drive/folders/1dOnUqtVB8Zfoume3oGjAbmFhMFTOin_I?usp=share_link) and put it in correct folder.
# Quick Start 
You can modify the inference.py to choose the datasets you want. (e.g.,   
    dataset = MP3DDataset(root_dir='/opt/data/private/360Layout/Datasets/mp3d', mode='test')  
    #dataset = ZindDataset(root_dir='/opt/data/private/360Layout/Datasets/zind', mode='test')  
    #dataset = PanoS2D3DMixDataset(root_dir='/opt/data/private/360Layout/Datasets/pano_s2d3d', mode='test', subset='pano')  
    #dataset = PanoS2D3DMixDataset(root_dir='/opt/data/private/360Layout/Datasets/pano_s2d3d', mode='test', subset='s2d3d'))  
```
python inference.py --cfg src/my_config/mp3d.yaml --output_dir src/output/mp3d
```
