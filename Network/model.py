import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Layers.layers import PretrainedEncoder, DeeplabDecoder

class ULab(nn.Module):
    """ This is the base class for a multi task neural network for both road segmentation
        and traffic object detection. It is an encoder-decoder deep convolutional
        neural network.

        The encoder first extracts feature maps that are then sepearately passed on to a
        decoder for segmentation and object detection. All operations are implemented
        independent of input resolution. 

        The archticture is modular which allows for quick alterations to both the encoder 
        as well as the decoder modulse.


        Args:
            phase (string):                         either "test" or "train"
            encoder_config:                         array that defines feature map channel numbers
            segmentation_decoder_config:            array that defines segmen. feature channels
    """

    def __init__(self, 
                name,
                encoder_config,
                segmentation_decoder_config,
                debug = False):
        super(ULab, self).__init__()
        self.name = name
        self.decoder_config = segmentation_decoder_config
        self.encoder = PretrainedEncoder(*encoder_config[0])
        encoder_channels = self.encoder.get_source_depth()
        decoder_cfg = encoder_channels + self.decoder_config[1:]
        self.segmentation_decoder = DeeplabDecoder(*decoder_cfg, self.encoder.get_source_scale())
        self.training = False
        self.low_auxiliary = nn.Conv2d(encoder_channels[0], 19, 1)
        self.mid_auxiliary = nn.Conv2d(encoder_channels[1], 19, 1)

        if debug:
            print('Encoder Config:')
            for l in encoder_config:
                print(l)
            print('Decoder Config:')
            for l in self.decoder_config:
                print(l)
            self.summary()
        
    def summary(self):
        print(self)
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('The model has a total of {} trainable parameters'.format(params))
        model_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('The Encoder has a total of {} trainable parameters'.format(params))
        model_parameters = filter(lambda p: p.requires_grad, self.segmentation_decoder.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('The Decoder has a total of {} trainable parameters'.format(params))

    def forward(self, x):
        sources = self.encoder(x)
        out = self.segmentation_decoder(*sources)
        if self.training:
            aux1 = self.low_auxiliary(sources[0])
            aux2 = self.mid_auxiliary(sources[1])
            return out, aux1, aux2
        return out

def build_ULab(ulab_config, debug = True):
    """ Builds the network given a ULab-Config
        args:
            ulab_config : configuration object (see config.py)
    """
    net = ULab(ulab_config.get_name(),
                *(ulab_config.get_layouts()), debug)

    return net



