import json
import os
import sys
import argparse
import time
import random

sys.path.append('./lib')

import math
import numpy as np
from PIL import Image
import cv2
import torch
from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.rpn.bbox_transform import bbox_transform_inv
from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
from model.faster_rcnn.vgg16 import vgg16
from model.utils.blob import im_list_to_blob
from tqdm import tqdm

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

class HOD:
    def __init__(self, args):
        
        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs)
            
        self.args = args    
        
        cfg.USE_GPU_NMS = args.cuda
        np.random.seed(cfg.RNG_SEED)
        
        # load model
        model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
        if not os.path.exists(model_dir):
            raise Exception('There is no input directory for loading network from ' + model_dir)
        load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 
        
        # initilize the network here.
        if args.net == 'vgg16':
            self.fasterRCNN = vgg16(self.pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            self.fasterRCNN = resnet(self.pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            self.fasterRCNN = resnet(self.pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            
        self.fasterRCNN.create_architecture()
        
        print("load checkpoint %s" % (load_name))
        
        checkpoint = torch.load(load_name)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        self.fasterRCNN.to('cuda')

        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')
        
    def detect(self, input_img):
        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1).cuda()
        im_info = torch.FloatTensor(1).cuda()
        num_boxes = torch.LongTensor(1).cuda()
        gt_boxes = torch.FloatTensor(1).cuda()
        box_info = torch.FloatTensor(1) 
        
        with torch.no_grad():
            self.fasterRCNN.eval()

            thresh_hand = self.args.thresh_hand 
            thresh_obj = self.args.thresh_obj
            vis = self.args.vis
            
            im = input_img[:, :, ::-1]
            blobs, im_scales = _get_image_blob(im)
            
            assert len(im_scales) == 1, "Only single-image batch implemented"
            
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)
            
            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_() 
            
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extact predicted params
            contact_vector = loss_list[0][0] # hand contact state info
            offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach() # hand side info (left/right)

            # get hand contact 
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side 
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()
            
            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    if self.args.class_agnostic:
                        if self.args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            if vis:
                im2show = np.copy(im)
            obj_dets, hand_dets = None, None
            for j in range(1, len(self.pascal_classes)):
                # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                if self.pascal_classes[j] == 'hand':
                    inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
                elif self.pascal_classes[j] == 'targetobject':
                    inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if self.pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    if self.pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()
            
            if vis:
                im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)
            else:
                im2show = None
    
        return obj_dets, hand_dets, thresh_obj, thresh_hand, im2show