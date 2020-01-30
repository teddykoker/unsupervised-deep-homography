# Unsupervised Deep Homography - *Unoffical* PyTorch Implementation

[**Unsupervised Deep Homography: A Fast and Robust Homography Estimation
Model**](https://arxiv.org/abs/1709.03966)<br>
Ty Nguyen, Steven W. Chen, Shreyas S. Shivakumar, Camillo J. Taylor, Vijay
Kumar<br>


![models](figures/models.jpg)
*Figure from original paper. Proposed model is \(c\)*

## Implementation Details

This implementation leverages [kornia](https://github.com/kornia/kornia), an
open source differentiable computer vision library. Kornia is used for computing
the direct linear transform (DLT) as well as the spatial transformation.

`model.py`: Regression model implementation <br>
`dataset.py`: Synthetic data generator <br>
`train.py`: Train unsupervised model using photometric loss outlined in paper


### Training

The model can be trained using synthetic data, created from the
[COCO](http://cocodataset.org/) dataset. 

```bash
python train.py \
    --bs=128 \            # batch size
    --lr=3e-5 \          # learning rate
    --epochs=200 \       # number of epochs
    path/to/COCO/train/  # COCO training data
```
