# StyleGAN-Keras
A reimplementation StyleGAN in Keras

## Idea and progress
+ WGAN-GP baseline model
+ Mapping network Z to W
+ Inconsequence noise
+ Adaptive instance normalization
+ StyleGAN2's new structure (Replaced for Progressive Training)

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

options:
-b, --batch_size
    Number of generated images
-m, --model-path
    Path to trained model
-mode, --mode
    Visualization mode, 0 for static, 1 for interpolation
-b1, --beta_1
    Degree of Feature Variation
-b2, --beta_2
    Degree of Style Variation
```
