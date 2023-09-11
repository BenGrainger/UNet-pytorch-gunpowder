#%%
from gunpowder import *
import math
import numpy as np
import torch
from mknet import create_affinity_model
from unet import WeightedMSELoss
from gunpowder.torch import Train
from tqdm import tqdm
from utility_funcs import imshow

# training data
parent_dir = r'U:\users\beng\automatic_segmentation\LSD\data3D'
data_dir_list = ['trainA.zarr', 'trainB.zarr']

# Array keys for gunpowder interface
raw = ArrayKey('RAW')
labels = ArrayKey('LABELS')
gt_affs = ArrayKey('GT_AFFS')
affs_weights = ArrayKey('AFFS_WEIGHTS')
pred_affs = ArrayKey('PRED_AFFS')

# data parameters
x, y, z = 84, 268, 268
input_shape = Coordinate((x, y, z))
voxel_size = Coordinate((40, 4, 4)) 

# model parameters
batch_size = 1
in_channels= 1
num_fmaps= 3
fmap_inc_factor= 6
downsample_factors=[(1,2,2),(1,2,2),(1,3,3)]
lr=0.5e-4

# model load + save locations
log_dir = 'models\model4\log'
checkpoint_basename = 'models\model4\checkpoints'

def train_until(max_iteration):

    # create model and optimizer
    model, optimizer = create_affinity_model(in_channels, num_fmaps, fmap_inc_factor, downsample_factors, lr)

    # run forward pass with data shape to return output shape
    model_input = torch.ones([batch_size, in_channels, x, y, z])
    outputs = model(model_input)
    output_shape = Coordinate((outputs.shape[2], outputs.shape[3], outputs.shape[4]))


    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    # this is the loader of datasources into the pipeline
    sources = tuple(
            # read batches from the Zarr file
            ZarrSource(
                parent_dir+'/'+s,
                datasets = {
                    raw: '/raw',
                    labels: '/segmentation'
                },
                array_specs = {
                    raw: ArraySpec(interpolatable=True),
                    labels: ArraySpec(interpolatable=False)
                }
            ) +

            # convert raw to float in [0, 1]
            Normalize(raw) +

            # chose a random location for each requested batch
            RandomLocation()

            for s in data_dir_list
        ) 

    pipeline = sources

    # randomly choose a sample from our tuple of samples - this is absolutely necessary to be able to get any data!
    pipeline += RandomProvider()

    # add steps to the pipeline
    #randomly mirror and transpose a batch
    pipeline == SimpleAugment()

    # elastcally deform the batch
    pipeline += ElasticAugment(
        [4,40,40],
        [0,2,2],
        [0,math.pi/2.0],
        prob_slip=0.05,
        prob_shift=0.05,
        max_misalign=25)

    # randomly shift and scale the intensities
    pipeline += IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1,
        z_section_wise=True)

    # dilate the boundaries between labels
    pipeline += GrowBoundary(labels, 
                             steps=3,
                             only_xy=True)

    pipeline += AddAffinities(
        affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        labels=labels,
        affinities=gt_affs)

    # no longer need labels since we use the gt affs for training

    # create scale array to balance class losses (will then use the affs_weights array during training)
    pipeline += BalanceLabels(
            gt_affs,
            affs_weights)

    pipeline += Unsqueeze([raw])

    pipeline += Stack(batch_size)

    # add the model to the pipeline
    pipeline += Train(
            model,
            WeightedMSELoss(),
            optimizer,
            inputs={
                'input': raw
            },
            outputs={
                0: pred_affs
            },
            loss_inputs={
                0: pred_affs,
                1: gt_affs,
                2: affs_weights
            },
            checkpoint_basename=checkpoint_basename,
            save_every=10,
            log_every=1,
            log_dir=log_dir)


    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)


    # iterate through the pipeline with build to request a batch
    with build(pipeline):
        progress = tqdm(range(max_iteration))
        for i in progress:
            batch = pipeline.request_batch(request)
            
            if i % 5 == 0:
                start = request[labels].roi.get_begin()/voxel_size
                end = request[labels].roi.get_end()/voxel_size

                batch_raw = batch[raw].data[:,:,start[0]:end[0],start[1]:end[1],start[2]:end[2]]
                batch_raw_images = batch_raw[0][0][0:5]
                imshow(
                    raw=batch_raw_images
                )

                batch_labels = batch[labels].data
                batch_labels_images = batch_labels[0][0:5]
                imshow(
                    ground_truth=batch_labels_images
                )

                batch_affs = batch[gt_affs].data
                batch_affs_images = batch_affs[0][0][0:5]
                imshow(
                    target=batch_affs_images, target_name='target 1'
                )
                
                batch_pred = batch[pred_affs].data
                batch_pred_images = batch_pred[0][0][0:5]
                imshow(
                    prediction=batch_pred_images, prediction_name='prediction 1'
                )

                batch_affs = batch[gt_affs].data
                batch_affs_images = batch_affs[0][1][0:5]
                imshow(
                    target=batch_affs_images, target_name='target 2'
                )
                
                batch_pred = batch[pred_affs].data
                batch_pred_images = batch_pred[0][1][0:5]
                imshow(
                    prediction=batch_pred_images, prediction_name='prediction 2'
                )

                batch_affs = batch[gt_affs].data
                batch_affs_images = batch_affs[0][2][0:5]
                imshow(
                    target=batch_affs_images, target_name='target 3'
                )
                
                batch_pred = batch[pred_affs].data
                batch_pred_images = batch_pred[0][2][0:5]
                imshow(
                    prediction=batch_pred_images, prediction_name='prediction 3'
                )




if __name__ == '__main__':

    iterations = 3000
    train_until(iterations)

# %%

