from __future__ import print_function
import argparse
import logging

import mxnet as mx

from rcnn.config import config, default, generate_config
from rcnn.tools.train_rpn import train_rpn
from rcnn.tools.test_rpn import test_rpn
from rcnn.tools.train_rcnn import train_rcnn
from rcnn.utils.combine_model import combine_model


def alternate_train(args, ctx, pretrained, epoch,
                    rpn_epoch, rpn_lr, rpn_lr_step,
                    rcnn_epoch, rcnn_lr, rcnn_lr_step):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # basic config
    begin_epoch = 0
    config.TRAIN.BG_THRESH_LO = 0.0

    logging.info('########## TRAIN RPN WITH IMAGENET INIT')
    train_rpn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
              args.frequent, args.kvstore, args.work_load_list, args.no_flip, args.no_shuffle, args.resume,
              ctx, pretrained, epoch, 'model/rpn1', begin_epoch, rpn_epoch,
              train_shared=False, lr=rpn_lr, lr_step=rpn_lr_step,
              use_data_augmentation=args.use_data_augmentation)

    logging.info('########## GENERATE RPN DETECTION')
    image_sets = [iset for iset in args.image_set.split('+')]
    for image_set in image_sets:
        test_rpn(args.network, args.dataset, image_set, args.root_path, args.dataset_path,
                 ctx[0], 'model/rpn1', rpn_epoch,
                 vis=False, shuffle=False, thresh=0)

    logging.info('########## TRAIN RCNN WITH IMAGENET INIT AND RPN DETECTION')
    train_rcnn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
               args.frequent, args.kvstore, args.work_load_list, args.no_flip, args.no_shuffle, args.resume,
               ctx, pretrained, epoch, 'model/rcnn1', begin_epoch, rcnn_epoch,
               train_shared=False, lr=rcnn_lr, lr_step=rcnn_lr_step, proposal='rpn',
               use_data_augmentation=args.use_data_augmentation, use_global_context=args.use_global_context,
               use_roi_align=args.use_roi_align)

    logging.info('########## TRAIN RPN WITH RCNN INIT')
    train_rpn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
              args.frequent, args.kvstore, args.work_load_list, args.no_flip, args.no_shuffle, args.resume,
              ctx, 'model/rcnn1', rcnn_epoch, 'model/rpn2', begin_epoch, rpn_epoch,
              train_shared=True, lr=rpn_lr, lr_step=rpn_lr_step, use_data_augmentation=args.use_data_augmentation)

    logging.info('########## GENERATE RPN DETECTION')
    image_sets = [iset for iset in args.image_set.split('+')]
    for image_set in image_sets:
        test_rpn(args.network, args.dataset, image_set, args.root_path, args.dataset_path,
                 ctx[0], 'model/rpn2', rpn_epoch,
                 vis=False, shuffle=False, thresh=0)

    logger.info('########## COMBINE RPN2 WITH RCNN1')
    combine_model('model/rpn2', rpn_epoch, 'model/rcnn1', rcnn_epoch, 'model/rcnn2', 0)

    logger.info('########## TRAIN RCNN WITH RPN INIT AND DETECTION')
    train_rcnn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
               args.frequent, args.kvstore, args.work_load_list, args.no_flip, args.no_shuffle, args.resume,
               ctx, 'model/rcnn2', 0, 'model/rcnn2', begin_epoch, rcnn_epoch,
               train_shared=True, lr=rcnn_lr, lr_step=rcnn_lr_step, proposal='rpn',
               use_data_augmentation=args.use_data_augmentation, use_global_context=args.use_global_context,
               use_roi_align=args.use_roi_align)

    logger.info('########## COMBINE RPN2 WITH RCNN2')
    combine_model('model/rpn2', rpn_epoch, 'model/rcnn2', rcnn_epoch, 'model/final', 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--no_flip', help='disable flip images', action='store_true')
    parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
    parser.add_argument('--resume', help='continue training', action='store_true')
    # alternate
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--rpn_epoch', help='end epoch of rpn training', default=default.rpn_epoch, type=int)
    parser.add_argument('--rpn_lr', help='base learning rate', default=default.rpn_lr, type=float)
    parser.add_argument('--rpn_lr_step', help='learning rate steps (in epoch)', default=default.rpn_lr_step, type=str)
    parser.add_argument('--rcnn_epoch', help='end epoch of rcnn training', default=default.rcnn_epoch, type=int)
    parser.add_argument('--rcnn_lr', help='base learning rate', default=default.rcnn_lr, type=float)
    parser.add_argument('--rcnn_lr_step', help='learning rate steps (in epoch)', default=default.rcnn_lr_step, type=str)
    # tricks
    parser.add_argument('--use_global_context', help='use roi global context for classification', action='store_true')
    parser.add_argument('--use_data_augmentation',
                        help='randomly transform image in color, brightness, contrast, sharpness',\
                        action='store_true')
    parser.add_argument('--use_roi_align', help='replace ROIPooling with ROIAlign', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    alternate_train(args, ctx, args.pretrained, args.pretrained_epoch,
                    args.rpn_epoch, args.rpn_lr, args.rpn_lr_step,
                    args.rcnn_epoch, args.rcnn_lr, args.rcnn_lr_step)

if __name__ == '__main__':
    main()
