import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    """ This is a block that utilizes Dilated convolutions to extract features at multiple scales from a feature map
        In the forward pass, the input gets (in parallel) passed into 4 convolutional layers with different receptive fields.
        The difference comes from using different values of diliation, i.e. 'distance' of kernel elements to each other.
        args:
            in_channels, mid_channels:              number of channels respectively.
            dilations:                              dilation rates used.
            separable:                              use separable convolutions.
    """
    def __init__(self, in_channels, mid_channels, dilations = [6, 12, 18], separable = False):
        super(ASPP, self).__init__()
        self.conv_1 = ConvBatchnorm(in_channels, mid_channels, 1, separable=separable)
        self.conv_2 = ConvBatchnorm(in_channels, mid_channels, 3, 1, dilations[0], separable)
        self.conv_3 = ConvBatchnorm(in_channels, mid_channels, 3, 1, dilations[1], separable)
        self.conv_4 = ConvBatchnorm(in_channels, mid_channels, 3, 1, dilations[2], separable)
        self._init_weight()

    def forward(self, x):
        s1 = self.conv_1(x)
        s2 = self.conv_2(x)
        s3 = self.conv_3(x)
        s4 = self.conv_4(x)
        s = torch.cat((s1, s2, s3, s4), dim = 1)
        return s

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class DenseUpsamplingConvolution(nn.Module):
    """ This block provides DeepLab-Sytle dense upsampling convolution

        args:
            in_channels:                number of input channels
            out_channels:               output_channels
            upsample_factor:            scaling factor
            no_activation:              set to true if no activation function should be used
    """
    def __init__(self, in_channels, out_channels, upsample_factor, no_activation = False):
        super(DenseUpsamplingConvolution, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, (upsample_factor**2) * out_channels, kernel_size=3, padding = 1) if no_activation else ConvBatchnorm(in_channels, upsample_factor**2 * out_channels, kernel_size=3, no_activation=True)
        self.shuffle = nn.PixelShuffle(upsample_factor)
        self._init_weight()
    def forward(self, x):
        x = self.conv_in(x)
        x = self.shuffle(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BilinearUpsample(nn.Module):
    """ Simple bilinear upsampling block

        args:
            upsample_factor:            scaling factor
    """
    def __init__(self, upsample_factor):
        super(BilinearUpsample, self).__init__()
        self.upsample_factor = upsample_factor
    
    def forward(self,x):
        return F.interpolate(x, scale_factor = self.upsample_factor, mode = 'bilinear', align_corners = False)

class ConvBatchnorm(nn.Module):
    ''' Simple building block that unifies a convolution, activation and batch normalization. 

        args:
            in_channels:        number of input channels
            out_channels:       number of output channels
            kernel_size:        kernel size
            dilation:           dilation rate of kernel
            separable:          bool indicating if separable convolution should be used
            no_activation:      set to true to leave out batchnorm and activation function
    '''
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1, separable = False, no_activation = False):
        super(ConvBatchnorm, self).__init__()
        if not isinstance(kernel_size, tuple):
            padding = (kernel_size // 2)*dilation
        else:
            padding = ((kernel_size[0] // 2)*dilation, (kernel_size[1] // 2)*dilation)
        if not no_activation:
            if not separable:
                self.map = nn.Sequential(nn.BatchNorm2d(in_channels, momentum=0.001), nn.LeakyReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias = False))
            else:
                self.map = nn.Sequential(nn.BatchNorm2d(in_channels, momentum=0.001), nn.LeakyReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=min(out_channels, in_channels), bias = False), nn.Conv2d(out_channels,out_channels, kernel_size=1, bias = False))
        else:
            if not separable:
                self.map = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias = False)
            else:
                self.map = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=min(out_channels, in_channels), bias = False), nn.Conv2d(out_channels,out_channels, kernel_size=1, bias = False))
        

    def forward(self, x):
        return self.map(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _init_zero(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)


class DeeplabDecoder(nn.Module):
    """ Deeplab style decoder for semantic segmentation. Gets input from multiple encoder scale levels as indicated by the scales parameter. Concatenates and upsamples to original size for prediction.

        args:
            low_level_in_channels, mid_level_in_channels, aspp_in_channels:                     channel depth of encoder output features
            low_level_channels, mid_level_channels, aspp_channels:                              desired channel depth after channel reduction/increase
            out_conv_channels:                                                                  number of channels of final upscaling and convolutions
            num_classes:                                                                        number of output classes
            scales:                                                                             spatial scale of input feature maps, e.g. 4, 8 etc. for different stages of downscaling in the encoder.
    """
    def __init__(self, low_level_in_channels, mid_level_in_channels, aspp_in_channels, low_level_channels, mid_level_channels, aspp_channels, out_conv_channels, num_classes, scales = [4, 8, 16]):
        super(DeeplabDecoder, self).__init__()
        
        self.conv_in_low = nn.Sequential(nn.Dropout(p=0.2), ConvBatchnorm(low_level_in_channels, low_level_channels, 1))
        self.conv_in_mid = nn.Sequential(nn.Dropout(p=0.2), ConvBatchnorm(mid_level_in_channels, mid_level_channels, 1))
        self.conv_in_aspp = nn.Sequential(nn.Dropout(p=0.2), ConvBatchnorm(aspp_in_channels, aspp_channels, 1))
        self.aspp_upsample = BilinearUpsample(scales[-1]//scales[0])
        self.mid_upsample = BilinearUpsample(scales[1]//scales[0])
        self.conv_out = nn.Sequential(ConvBatchnorm(mid_level_channels + low_level_channels + aspp_channels, out_conv_channels, 3), ConvBatchnorm(out_conv_channels, out_conv_channels, 1, no_activation=False), DenseUpsamplingConvolution(out_conv_channels, num_classes, scales[0]))

    def forward(self, low, mid, aspp):
        low = self.conv_in_low(low)
        mid = self.conv_in_mid(mid)
        mid = self.mid_upsample(mid)
        aspp = self.conv_in_aspp(aspp)
        aspp = self.aspp_upsample(aspp)
        combined_features = torch.cat((low, mid, aspp), dim = 1)
        out = self.conv_out(combined_features)
        return out

class PretrainedEncoder(nn.Module):
    """ Simple encoder that extracts the convolutional part of pre-trained PyTorch model zoo networks. 
        Automatically computes scale levels and saves channel depths during construction. Source layers are currently hardcoded to correspond with the quarter and eighth scale levels, but feel free to experiment.

        args:
            base_string:                    string indicating which classifier to use, supports: wide_resnet50, resnet50, resnet34, resnet18, resnext50, densenet121, mobilenet, vgg16_bn and vgg16. 

    """
    def __init__(self, base_string, aspp_channels, pre_trained = True):
        super(PretrainedEncoder, self).__init__()
        if base_string == 'wide_resnet50':
            base = models.wide_resnet50_2(pretrained=pre_trained)
            base_list = list(base.children())[:-2]
            source_layers = [4, 5]
        elif base_string == 'resnet50': 
            base = models.resnet50(pretrained=pre_trained)
            base_list = list(base.children())[:-2]
            source_layers = [4, 5]
        elif base_string == 'resnet34': 
            base = models.resnet34(pretrained=pre_trained)
            base_list = list(base.children())[:-2]
            source_layers = [4, 5]
        elif base_string == 'resnet18': 
            base = models.resnet18(pretrained=pre_trained)
            base_list = list(base.children())[:-2]
            source_layers = [4, 5]
        elif base_string == 'resnext50': 
            base = models.resnext50_32x4d(pretrained=pre_trained)
            base_list = list(base.children())[:-2]
            source_layers = [4, 5]
        elif base_string == 'densenet121': 
            base = models.densenet121(pretrained=pre_trained).features
            base_list = list(base.children())
            source_layers = [4, 6]
        elif base_string == 'mobilenet': 
            base = models.mobilenet_v2(pretrained=pre_trained).features
            base_list = list(base.children())[:-1]
            source_layers = [3, 6]
        elif base_string == 'vgg16_bn': 
            base = models.vgg16_bn(pretrained=pre_trained).features
            base_list = list(base.children())[:-1]
            source_layers = [12, 22]
        elif base_string == 'vgg16': 
            base = models.vgg16(pretrained=pre_trained).features
            base_list = list(base.children())[:-1]
            source_layers = [8, 15]
        
        self.layers = nn.ModuleList(base_list)
        self.source_layers = source_layers

        # Feed example inputs through the net to get output channel depths!
        self.source_depth = list()
        x = torch.rand((1, 3, 64, 32))
        self.source_scale = []
        for k, l in enumerate(self.layers):
            x = l(x)
            if k in self.source_layers:
                self.source_depth.append(x.size()[1]) 
                self.source_scale.append(64//x.size()[2])
        
        self.aspp = ASPP(x.size()[1], aspp_channels)
        x = self.aspp(x)
        self.source_depth.append(x.size()[1])
        self.source_scale.append(64//x.size()[2])

    def get_source_depth(self):
        return self.source_depth 
    
    def get_source_scale(self):
        return self.source_scale

    def forward(self, x):
        out = []
        for k, l in enumerate(self.layers):
            x = l(x)
            if k in self.source_layers:
                out.append(x)
        x = self.aspp(x)
        out.append(x)
        return out

if __name__ == "__main__":
    pass
