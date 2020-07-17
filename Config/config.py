class ULabConfig(object):
    """ Just a simple organizer class for initializing the network, basically a remnant of the much larger multi-task architecture I'm working on.
        TODO: Refactor / get rid of
    """
    def __init__(self, cfg):
        super(ULabConfig, self).__init__()
        self.encoder_layout = GetEncoderLayout(cfg)
        self.segmentation_decoder_layout = GetDecoderLayout(cfg)
        self.name = cfg['name']

    def get_layouts(self):
        return [self.encoder_layout, self.segmentation_decoder_layout]
    def get_name(self):
        return self.name
    


config = {   
    'ULab_resnet34': {
        'classes' : [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        'input_size' : (2048,1024),
        'name' : 'ULab_resnet34',

        'encoder' : 'resnet34',
        'decoder_channels' : [128, 128],
        'decoder' : 'ULab',
        'aspp_channels' : [64, 128],
        'conv_out_channels' : 128
    },

    'ULab_resnet18': {
        'classes' : [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        'input_size' : (2048,1024),
        'name' : 'ULab_resnet18',

        'encoder' : 'resnet18',
        'decoder_channels' : [128, 128],
        'decoder' : 'ULab',
        'aspp_channels' : [64, 128],
        'conv_out_channels' : 128
    },

    'ULab_resnet50': {
        'classes' : [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        'input_size' : (2048,1024),
        'name' : 'ULab_resnet50',

        'encoder' : 'resnet50',
        'decoder_channels' : [128, 128],
        'decoder' : 'ULab',
        'aspp_channels' : [64, 128],
        'conv_out_channels' : 128
    },

    'ULab_wide_resnet50': {
        'classes' : [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        'input_size' : (2048,1024),
        'name' : 'ULab_wide_resnet50',

        'encoder' : 'wide_resnet50',
        'decoder_channels' : [128, 128],
        'decoder' : 'ULab',
        'aspp_channels' : [64, 128],
        'conv_out_channels' : 128
    },

    'ULab_mobilenet': {
        'classes' : [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        'input_size' : (2048,1024),
        'name' : 'ULab_mobilenet',

        'encoder' : 'mobilenet',
        'decoder_channels' : [128, 128],
        'decoder' : 'ULab',
        'aspp_channels' : [64, 128],
        'conv_out_channels' : 128
    },

    'ULab_resnext50': {
        'classes' : [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        'input_size' : (2048,1024),
        'name' : 'ULab_resnext50',

        'encoder' : 'resnext50',
        'decoder_channels' : [128, 128],
        'decoder' : 'ULab',
        'aspp_channels' : [64, 128],
        'conv_out_channels' : 128
    },

    'ULab_densenet': {
        'classes' : [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        'input_size' : (2048,1024),
        'name' : 'ULab_densenet',

        'encoder' : 'densenet121',
        'decoder_channels' : [128, 128],
        'decoder' : 'ULab',
        'aspp_channels' : [64, 128],
        'conv_out_channels' : 128
    },
}

def GetEncoderLayout(cfg):
    aspp_channels = cfg.get('aspp_channels')
    return [
                [cfg['encoder'], aspp_channels[0]],
            ]
         
def GetDecoderLayout(cfg):
    decoder_channels = cfg.get('decoder_channels')
    aspp_channels = cfg.get('aspp_channels')
    conv_out_channels = cfg.get('conv_out_channels')
    return ['DeeplabDecoder', decoder_channels[0], decoder_channels[1], aspp_channels[1], conv_out_channels, len(cfg['classes'])]
            
    