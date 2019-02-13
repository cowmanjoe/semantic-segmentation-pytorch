# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
import h5py


def convert_to_segmented(segmentation_module, image, args):
    segmentation_module.eval()

    # TODO: make sure dimension ordering is correct
    segSize = (image.shape[2], image.shape[1])

    with torch.no_grad():
        # scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        # scores = async_copy_to(scores, args.gpu)

        feed_dict = {'img_data': image.unsqueeze(0)}
        feed_dict = async_copy_to(feed_dict, args.gpu)

        scores = segmentation_module(feed_dict, segSize=segSize)

        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

    segmented = blackout(image.permute(1, 2, 0), pred)

    return segmented

def build_segmentation_module(args):
    # torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    # list_test = [{'fpath_img': args.test_img}]
    # list_test = [{'fpath_img': x} for x in args.test_imgs]
    # dataset_test = TestDataset(
    #     list_test, args, max_sample=args.num_val)
    # loader_test = torchdata.DataLoader(
    #     dataset_test,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=user_scattered_collate,
    #     num_workers=5,
    #     drop_last=True)

    return segmentation_module

def batch_segment_images(args):
    seg_module = build_segmentation_module(args)

    train_images = None
    valid_images = None

    with h5py.File(args.h5_path, 'r') as f:
        train_images = np.array(f['train_img'][:10], dtype='u1')
        valid_images = np.array(f['valid_img'][:10], dtype='u1')

    for i in range(train_images.shape[0]):
        train_images[i] = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2RGB)

    for i in range(valid_images.shape[0]):
        valid_images[i] = cv2.cvtColor(valid_images[i], cv2.COLOR_BGR2RGB)

    train_images = torch.tensor(train_images, dtype=torch.float32)
    valid_images = torch.tensor(valid_images, dtype=torch.float32)

    # TODO: verify this permutes to NCHW
    train_images = train_images.permute(0, 3, 1, 2)
    valid_images = valid_images.permute(0, 3, 1, 2)

    seg_train_images = torch.zeros(train_images.shape, dtype=torch.float32)

    for img in train_images:
        segmented = convert_to_segmented(seg_module, img, args)

        visual = np.concatenate((img.permute(1, 2, 0), segmented), axis=1).astype(np.uint8)

        cv2.imshow('test', visual)
        cv2.waitKey()



# My function to blackout everything except the house
def blackout(img, pred):
    houselist = [0, 1, 3, 5, 6, 8, 11, 13, 14, 25, 29, 34, 42, 48, 52, 53, 54, 58, 59, 61, 70, 79, 86, 88, 93, 94, 95,
                 96, 114]
    # wall, building, floor, ceiling, road, windowpane, sidewalk, earth, door, house, field, rock, column, skyscraper, path, stairs, runway, screendoor, stairway, bridge, countertop, hovel, awning, booth, pole, land, banister, escalator, and tent
    black_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    it = np.nditer(pred, flags=['multi_index'])

    while not it.finished:
        if it[0] in houselist:
            black_img[it.multi_index[0], it.multi_index[1], :] = img[it.multi_index[0], it.multi_index[1], :]
        it.iternext()

    return black_img

def main(args):
    batch_segment_images(args)


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--h5_path', required=True,
                        help='path of the hdf5 file')
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')

    args = parser.parse_args()
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)

    assert os.path.exists(args.weights_encoder) and \
           os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)