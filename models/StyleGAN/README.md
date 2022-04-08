# StyleGAN-Keras
A customized StyleGAN integrated StyleGAN 2 network architecture

## Idea and progress
+ WGAN-GP baseline model
+ Mapping network Z to W
+ Inconsequence noise
+ Adaptive instance normalization
+ Progressive Training (Replaced by StyleGAN2's new structure)

## Recommmended Dataset
Danbooru dataset of portrait faces processed by GWERN: https://www.gwern.net/BigGAN

## Generated samples from StyleGAN
![StyleGAN_Preview](https://raw.githubusercontent.com/akn0717/Anime-Character-Face-Generator-Keras/master/StyleGANPreview.jpg)

## Frameworks
Keras, Tensorflow

## Instructions

- Install dependencies
```bash
pip install -r requirements.txt
```

- Generate random anime faces
```bash
python generate.py [-options]
```
