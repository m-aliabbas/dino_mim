from PIL import Image
import numpy as np

import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

from numpy.random import randint
import io
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms as tf
from typing import Optional, Tuple

from torchvision import transforms


def buildLabelIndex(labels):
    """
    This function builds an index for labels. It creates a dictionary where the keys are the unique labels 
    and the values are lists of indices where each label appears in the input list.

    Args:
        labels (list): A list of labels.

    Returns:
        dict: A dictionary with labels as keys and lists of indices as values.
    """
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def getItem(idx, X, target = None, transform=None, training_mode = 'SSL'):
    """
    This function applies a transformation to the input data if a transform function is provided and returns the data along with the target.

    Parameters:
    idx (int): Index of the item to be retrieved.
    X (array-like): Input data.
    target (array-like, optional): Target data. Default is None.
    transform (callable, optional): A function/transform that takes in an array-like and returns a transformed version. Default is None.
    training_mode (str, optional): Mode of training. Default is 'SSL'.

    Returns:
    tuple: Transformed input data and target.
    """
    if transform is not None:
        X = transform(X)

    return X, target

import torchvision.transforms as tf
from PIL import Image

class myRandCrop(tf.RandomResizedCrop):
    """
    Custom random resized crop transformation that returns the cropped image
    along with the crop coordinates.

    Args:
        size (int or tuple): Desired output size of the crop. If size is an int
            instead of tuple like (h, w), a square output size (size, size) is
            made.
        scale (tuple of float): Specifies the lower and upper bounds for the
            random area of the crop, before resizing. Default is (0.08, 1.0).
        ratio (tuple of float): Specifies the lower and upper bounds for the
            random aspect ratio of the crop, before resizing. Default is (3/4, 4/3).
        interpolation (int): Desired interpolation enum defined by `PIL.Image`.
            Default is `PIL.Image.BILINEAR`.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super(myRandCrop, self).__init__(size, scale, ratio, interpolation)
        
    def forward(self, img):
        """
        Apply the random resized crop transformation to the image and return
        the cropped image along with the crop coordinates.

        Args:
            img (PIL Image or Tensor): Input image to be cropped and resized.

        Returns:
            (PIL Image or Tensor, tuple): Tuple containing the cropped and resized image
            and a tuple with the crop coordinates (i, j, h, w).

            i,j is start,end
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)  # Get the parameters for the crop
        return tf.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)


class myRandomHorizontalFlip(tf.RandomHorizontalFlip):
    """
    Custom random horizontal flip transformation that returns the flipped image
    along with a flag indicating whether the image was flipped.

    Args:
        p (float): Probability of the image being flipped. Default is 0.5.
    """
    def __init__(self, p=0.5):
        super(myRandomHorizontalFlip, self).__init__(p=p)
        
    def forward(self, img):
        """
        Apply the random horizontal flip transformation to the image and return
        the flipped image along with a flag indicating whether the image was flipped.

        Args:
            img (PIL Image or Tensor): Input image to be possibly flipped.

        Returns:
            (PIL Image or Tensor, int): Tuple containing the possibly flipped image
            and an integer flag (1 if the image was flipped, 0 otherwise).
        """
        if torch.rand(1) < self.p:  # Draw a random number and check if it's less than p
            return tf.functional.hflip(img), 1  # Flip the image horizontally and return 1
        return img, 0  # Return the original image and 0

    
    
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.

    Args:
        p (float): Probability of applying the Gaussian Blur. Default is 0.5.
        radius_min (float): Minimum radius for the Gaussian Blur. Default is 0.1.
        radius_max (float): Maximum radius for the Gaussian Blur. Default is 2.0.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p  # Probability of applying the Gaussian Blur
        self.radius_min = radius_min  # Minimum radius for the blur
        self.radius_max = radius_max  # Maximum radius for the blur

    def __call__(self, img):
        """
        Apply Gaussian Blur to the image with a certain probability.

        Args:
            img (PIL Image): Input image to be possibly blurred.

        Returns:
            PIL Image: The possibly blurred image.
        """
        # Determine if the blur should be applied
        do_it = random.random() <= self.prob
        if not do_it:
            return img  # Return the original image if not applying the blur

        # Apply Gaussian Blur with a random radius between radius_min and radius_max
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.

    Args:
        p (float): Probability of applying solarization.
    """
    def __init__(self, p):
        self.p = p  # Probability of applying solarization

    def __call__(self, img):
        """
        Apply solarization to the image with a certain probability.

        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Solarized image if the probability condition is met, otherwise the original image.
        """
        # Check if solarization should be applied based on the probability
        if random.random() < self.p:
            return ImageOps.solarize(img)  # Apply solarization
        else:
            return img  # Return the original image




# def distort_images(samples, masks, drop_rep, drop_align):
    
#     B = samples.size()[0] 
#     samples_aug = samples.detach().clone()
#     for i in range(B):
#         idx_rnd = randint(0, B)
#         if idx_rnd != i:
#             samples_aug[i], masks[i] = replace_rand_patches(samples[i].detach().clone(), 
#                                                   X_rep = samples_aug[idx_rnd],
#                                                   mask = masks[i],
#                                                   max_replace=drop_rep, align=drop_align)
      
#     return samples_aug, masks



def GMML_replace_list(samples, corrup_prev, masks_prev, drop_type='noise', max_replace=0.35, align=16):
        
    rep_drop = 1 if drop_type == '' else ( 1 / ( len(drop_type.split('-')) + 1 ) )
    
    n_imgs = samples.size()[0] #this is batch size, but in case bad inistance happened while loading
    samples_aug = samples.detach().clone()
    masks = torch.zeros_like(samples_aug)
    for i in range(n_imgs):
        idx_rnd = randint(0, n_imgs)
        if random.random() < rep_drop: 
            samples_aug[i], masks[i] = GMML_drop_rand_patches(samples_aug[i], samples[idx_rnd], max_replace=max_replace, align=align)
        else:
            samples_aug[i], masks[i] = corrup_prev[i], masks_prev[i]

    return samples_aug, masks

def GMML_drop_rand_patches(X, X_rep=None, drop_type='noise', max_replace=0.7, align=16, max_block_sz=0.3):
    """
    Randomly drops patches in the input tensor X and replaces them with noise, zeros, or patches from X or X_rep.
    
    Args:
        X (torch.Tensor): Input tensor of shape (C, H, W) where C is the number of channels, H is the height, and W is the width.
        X_rep (torch.Tensor, optional): Replacement tensor of the same shape as X. If provided, patches from X_rep are used for replacement. Default is None.
        drop_type (str, optional): Type of drop replacement. Can be 'noise', 'zeros', or 'rand'. Default is 'noise'.
        max_replace (float, optional): Maximum percentage of the image to be replaced. Default is 0.7.
        align (int, optional): Alignment for the patch sizes. Default is 16.
        max_block_sz (float, optional): Maximum size of the block to be dropped as a percentage of the image size. Default is 0.3.
        
    Returns:
        torch.Tensor: Tensor with patches dropped and replaced.
        torch.Tensor: Mask tensor indicating which parts of the image were replaced.
    """
    np.random.seed()    
    C, H, W = X.size()  # Get the dimensions of the input tensor
    
    # Calculate the number of pixels to drop
    n_drop_pix = np.random.uniform(min(0.5, max_replace), max_replace) * H * W
    
    # Calculate the maximum block height and width to be dropped
    mx_blk_height = int(H * max_block_sz)
    mx_blk_width = int(W * max_block_sz)
    
    align = max(1, align)  # Ensure alignment is at least 1
    
    mask = torch.zeros_like(X)  # Initialize the mask tensor
    drop_t = np.random.choice(drop_type.split('-'))  # Choose the drop type
    
    
    while mask[0].sum() < n_drop_pix:
        
        ####### get a random block to replace 
        rnd_r = ( randint(0, H-align) // align ) * align
        rnd_c = ( randint(0, W-align) // align ) * align

        rnd_h = min(randint(align, mx_blk_height), H-rnd_r)
        rnd_h = round( rnd_h / align ) * align
        rnd_w = min(randint(align, mx_blk_width), W-rnd_c)
        rnd_w = round( rnd_w / align ) * align
        
        if X_rep is not None:
            X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X_rep[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w].detach().clone()
        else:
            if drop_t == 'noise':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.empty((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device).normal_()
            elif drop_t == 'zeros':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.zeros((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device)
            else:
                ####### get a random block to replace from
                rnd_r2 = (randint(0, H-rnd_h) // align ) * align
                rnd_c2 = (randint(0, W-rnd_w) // align ) * align
            
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X[:, rnd_r2:rnd_r2+rnd_h, rnd_c2:rnd_c2+rnd_w].detach().clone()
            
        mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1 
         
    return X, mask

    
class DataAugmentation(object):
    def __init__(self, args):
        
        # for corruption
        self.drop_perc = args.drop_perc
        self.drop_type = args.drop_type
        self.drop_align = args.drop_align
        
        global_crops_scale = args.global_crops_scale
        local_crops_scale = args.local_crops_scale
        global_crops_number = args.global_crops_number
        local_crops_number = args.local_crops_number
        
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])
        
    def corrupt_img(self, im):
        im_corrupted = im.detach().clone()
        im_mask = torch.zeros_like(im_corrupted)
        if self.drop_perc > 0:
            im_corrupted, im_mask = GMML_drop_rand_patches(im_corrupted, 
                                                           max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)
            return im, im_corrupted, im_mask
        else:
            return im, None, None
        

    def __call__(self, image):
        clean_crops = []
        corrupted_crops = []
        masks_crops = []
        
        im, im_corrupted, im_mask = self.corrupt_img(self.global_transfo1(image))
        clean_crops.append(im)
        corrupted_crops.append(im_corrupted)
        masks_crops.append(im_mask)
            
        im, im_corrupted, im_mask = self.corrupt_img(self.global_transfo2(image))
        clean_crops.append(im)
        corrupted_crops.append(im_corrupted)
        masks_crops.append(im_mask)
        
        for _ in range(self.local_crops_number):
            clean_crops.append(self.local_transfo(image))
            
        return clean_crops, corrupted_crops, masks_crops