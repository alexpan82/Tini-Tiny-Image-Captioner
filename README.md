# Tini-Tiny Image Captioner

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>   

 * Inspired by ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734) and a fork of [Mokady et al](https://github.com/rmokady/CLIP_prefix_caption).
 * See our [paper]() for detailed description of our project

## Description
Image captioning is an area of deep learning that is currently receiving massive interest. However, these systems and image captioning models in general tend to be very computationally expensive due to their massive frameworks with very deep neural networks and heavy decoders, encoders, and transformers. Trying to experiment with and extend these models is impossible for the average person with consumer-grade hardware. We tackled this problem by creating a smaller and lightweight image captioning system that can be trained and implemented end-to-end on a single GPU.

## Set up
### Environment
```
conda env create -f environment.yml
```
### Data download
- Download modified [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/coco/annotations`
- Download original captions as well in this way:
```
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
mv annotations annotations_org
```
- Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip.

## COCO training
Extract TinViT features using (output is `data/coco/oscar_split_tinyvit_train.pkl`):
```
python parse_coco.py --clip_model_type tinyvit
```

Train transformer mapping network:
```
python train.py --data ./data/coco/oscar_split_tinyvit_train.pkl --out_dir ./coco_train_transformer_4/ --mapping_type transformer --num_layers 8 --prefix_length 40 --prefix_length_clip 40
```
See [Mokady et al](https://github.com/rmokady/CLIP_prefix_caption) for how to use other `train.py` arguments

## Inference Notebooks
Use `notebooks/transformer_inference.ipynb` to perform inference and visualize results.

## Acknowledgments
This repository is forked from [Mokady et al](https://github.com/rmokady/CLIP_prefix_caption) and utilizes [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home)