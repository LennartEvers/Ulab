# ULab - Simple and light-weight Semantic Segmentation CNN

This is a simple Encoder-Decoder network for semantic segmentation implemented in PyTorch (1.5.0). It differs to most other implementations by using the PyTorch model zoo classifiers as encoders. The decoder can be chosen from a variety of models, with the decoder being similar to deeplab with an extra skip connection. With smaller encoders such as resnet18, resnet34 or mobilenet the network is pretty lightweight and achieves inference speed greater than 20fps on the full resolution cityscapes images. 

## Performance

Accuracy is measured by mean intersection over union on the Cityscapes validation split. Inference speed is measured by the mean frames per second during inference on 500 images. All numbers are obtained on a single Nvidia RTX2080Ti.

| Encoder       | mIOU(%) |  FPS |
|---------------|:-------:|-----:|
| mobilenet v2  |   74.5  | 26.6 |
| resnet18      |   74.3  | 27.7 |
| resnet34      |   76.4  | 20.6 |
| resnet50      |   77.0  | 13.1 |
| wide resnet50 |   77.3  |  9.2 |
| resnext50     |   77.1  | 12.2 |
| densenet121   |   76.6  | 13.4 |


## Installation and Prerequisites

The code was tested with Anaconda and Python 3.8. The code runs fine on CPU, but you should have a CUDA capable GPU for a better experience. After setting up the conda environment:

1. Clone this repository
```
git clone https://github.com/LennartEvers/Ulab.git
```

```
cd Ulab
```
2. Install dependencies: For PyTorch, see [pytorch.org](https://www.pytorch.org) for more details. 
install opencv (for the visualization)
```
pip install opencv-python
```

## Running the network
Run 
```
python demo.py --config ULab_resnet34 --visualize_preds True
```
This visualizes predictions of the resnet34 based model on a test video (Stuttgart_02) from the cityscapes benchmark dataset. Note that the visualization massively slows down frame rate. For other configurations have a look into Config/config.py


## License



