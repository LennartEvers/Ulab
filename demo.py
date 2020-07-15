from Network.model import build_ULab
from Config.config import config, ULabConfig
import argparse
import cv2 
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# From the Cityscapes Benchmark repo:
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (165,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# label id to color dict
cityscapes_label2color = { label.id : label.color for label in labels }

def build_id_converter(classes):
    """ builds a dict to map a selection of classes to ids 0 ... len(classes)
    """
    converter = dict()
    for k, classId in enumerate(classes):
        converter[k] = classId
    return converter

class Visualizer(object):
    """ This class somewhat efficiently visualizes a prediction mask for an input image. If you find a faster way without leaving python please message me!
        args:
            classes:        list of labelIds (!) of the classes to visualize 
    """
    def __init__(self, classes):
        self.converter = build_id_converter(classes)
        self.id2color = {classId : cv2.cvtColor(np.uint8([[cityscapes_label2color[self.converter[classId]]]]), cv2.COLOR_BGR2HSV)[0, :] for classId in self.converter.keys()}
        self.num_classes = len(self.converter.keys())
    def __call__(self, mask, image = None):
        k = np.array(list(self.id2color.keys()))
        v = np.array(list(self.id2color.values()))
        self.key_value_zip = zip(k,v)
        height, width = mask.shape
        outputimage = np.zeros((height, width, 3)).astype(np.uint8)
        for key,val in self.key_value_zip:
            outputimage[mask==key] = val
        if image is not None:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
            image[:,:,:2] = outputimage[:,:,:2]
            image[:,:,2] = image[:,:,2] // 2 + 128
            outputimage = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return outputimage

class predictor(object):
    """ A simple wrapper that handles network loading, prediction, gpu/cpu and tensor conversion.
        args:
            cfg:        the configuration dict
    """
    def __init__(self, cfg):
        CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
        CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)])
        self.CUDA = torch.cuda.is_available()
        net_cfg = ULabConfig(cfg)
        net = build_ULab(net_cfg)
        net = nn.DataParallel(net)
        if self.CUDA:
            net.load_state_dict(torch.load('Weights/' + cfg['encoder'] + '.pth'))
        else:
            net.load_state_dict(torch.load('Weights/' + cfg['encoder'] + '.pth', map_location=torch.device('cpu')))
        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            net.cuda()
        self.net = net.module
        self.net.eval()

    def __call__(self, image):
        with torch.no_grad():
            x = self.transform(image)
            x = x.unsqueeze(0)
            if self.CUDA:
                x = x.cuda()
            out = self.net(x)
            mask = torch.argmax(out[0].float(), dim = 0).cpu().numpy()
        return mask

def main(args):
    cudnn.fastest = True
    cfg = config[args.config]
    net = predictor(cfg)
    visualizer = Visualizer(cfg['classes'])
    input_scale = 2
    video = cv2.VideoCapture(args.video_path)
    print('Starting prediction on {} with ULab config {}.'.format(args.video_path, args.config))
    while True:
        (grabbed, image) = video.read()
        if not grabbed:
            break
        w, h, _ = image.shape
        # To speed up inference scale down the input images by choosing input scale > 1
        w, h = w//input_scale, h//input_scale
        # network needs dimensions to be divisible by 32
        image = cv2.resize(image, (h//32*32, w//32*32))
        prediction = net(image)
        if args.visualize_preds:
            image = visualizer(prediction, image)
        cv2.imshow('test', image)
        key = cv2.waitKey(1)
        if key == 27:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ULab for semantic segmentation')
    parser.add_argument('--config', default='ULab_resnet34',
                        type=str, help='Configuration to use', choices=config.keys())
    parser.add_argument('--video_path', default='./Videos/Test.mp4', type=str,
                        help='path to video file to predict on')
    parser.add_argument('--visualize_preds', default=True, type=bool,
                        help='output prediction masks or only input video frames')
    args = parser.parse_args()
    main(args)