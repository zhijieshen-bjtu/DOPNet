Pytorch implementation of ["Disentangling Orthogonal Planes for Indoor Panoramic Room Layout Estimation with Cross-Scale Distortion Awareness"](https://arxiv.org/abs/2303.00971) (CVPR'23)  
The project is reproducted based on LGT-Net and [PanoFormer](https://github.com/zhijieshen-bjtu/PanoFormer). 
# Update
The generating mask process can be available in generate_segmentation-mask.py.
# Trained Models 
Download the model at this [link](https://drive.google.com/drive/folders/1dOnUqtVB8Zfoume3oGjAbmFhMFTOin_I?usp=share_link) and put it in correct folder.
# Quick Start 
You can modify the inference.py to choose the datasets you want. (e.g.,  
```
    dataset = MP3DDataset(root_dir='/opt/data/private/360Layout/Datasets/mp3d', mode='test')  
    #dataset = ZindDataset(root_dir='/opt/data/private/360Layout/Datasets/zind', mode='test')  
    #dataset = PanoS2D3DMixDataset(root_dir='/opt/data/private/360Layout/Datasets/pano_s2d3d', mode='test', subset='pano')  
    #dataset = PanoS2D3DMixDataset(root_dir='/opt/data/private/360Layout/Datasets/pano_s2d3d', mode='test', subset='s2d3d')
```
)  
```
python inference.py --cfg src/my_config/mp3d.yaml --output_dir src/output/mp3d
```
If you want to test your own data, please modify main() in inference.py:
```
if __name__ == '__main__':
    logger = get_logger()
    args = parse_option()
    config = get_config(args)

    if 'cuda' in args.device and not torch.cuda.is_available():
        logger.info(f'The {args.device} is not available, will use cpu ...')
        config.defrost()
        args.device = "cpu"
        config.TRAIN.DEVICE = "cpu"
        config.freeze()

    model, _, _, _ = build_model(config, logger)
    os.makedirs(args.output_dir, exist_ok=True)
    img_paths = sorted(glob.glob(args.img_glob))

    inference()
```
And run:
```
python inference.py --cfg src/my_config/mp3d.yaml --img_glob src/demo/{demo}.png --output_dir src/output/test --post_processing manhattan
```

If you find our work useful, please consider citing： 
```
@InProceedings{Shen_2023_CVPR,
    author    = {Shen, Zhijie and Zheng, Zishuo and Lin, Chunyu and Nie, Lang and Liao, Kang and Zheng, Shuai and Zhao, Yao},
    title     = {Disentangling Orthogonal Planes for Indoor Panoramic Room Layout Estimation With Cross-Scale Distortion Awareness},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {17337-17345}
}
```

