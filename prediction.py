#%%
from gunpowder import *
import torch

from mknet import create_affinity_model
from gunpowder.torch import Predict
from utility_funcs import imshow
import numpy as np

checkpoint = 'models\model4\checkpoints_checkpoint_3000'

# data keys
raw = ArrayKey('RAW')
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

# create model
model, optimizer = create_affinity_model(in_channels, num_fmaps, fmap_inc_factor, downsample_factors, lr)

model_input = torch.ones([batch_size, in_channels, x, y, z])
outputs = model(model_input)
output_shape = Coordinate((outputs.shape[2], outputs.shape[3], outputs.shape[4]))


input_size = input_shape * voxel_size
output_size = output_shape * voxel_size

# request prediction batch
scan_request = BatchRequest()

scan_request.add(raw, input_size)
scan_request.add(pred_affs, output_size)

context = (input_size - output_size) / 2

# load raw data
source = ZarrSource(
    r'U:\users\beng\automatic_segmentation\LSD\data3D/val.zarr',
    {
        raw: '/raw'
    },
    {
        raw: ArraySpec(interpolatable=True)
    }
    )

with build(source):
    total_input_roi = source.spec[raw].roi
    total_output_roi = source.spec[raw].roi.grow(-context,-context)

# set model to eval mode
model.eval()

# add a predict node
predict = Predict(
    model=model,
    checkpoint=checkpoint,
    inputs = {
            'input': raw
    },
    outputs = {
        0: pred_affs})


scan = Scan(scan_request)

pipeline = source
pipeline += Normalize(raw)

# raw shape = h,w

pipeline += Unsqueeze([raw])

# raw shape = c,h,w

pipeline += Stack(batch_size)

# raw shape = b,c,h,w

pipeline += predict
pipeline += scan
pipeline += Squeeze([raw])

# raw shape = c,h,w
# pred_affs shape = b,c,h,w

pipeline += Squeeze([raw, pred_affs])

# raw shape = h,w
# pred_affs shape = c,h,w

predict_request = BatchRequest()

# this lets us know to process the full image. we will scan over it until it is done
predict_request.add(raw, total_input_roi.get_end())
predict_request.add(pred_affs, total_output_roi.get_end())


with build(pipeline):
    batch = pipeline.request_batch(predict_request)


#%%

img_dir = 'models\model4\imgs'

pred = batch[pred_affs].data
raw_data = batch[raw].data

batch_raw_images = raw_data[50:55]
imshow(
    raw=batch_raw_images, save_name=img_dir+'/'+'raw.png'
)

batch_pred_images = pred[0][50:55]
imshow(
    prediction=batch_pred_images, prediction_name='prediction 1', save_name=img_dir+'/'+'prediction 1.png'
)

batch_pred_images = pred[1][50:55]
imshow(
    prediction=batch_pred_images, prediction_name='prediction 2', save_name=img_dir+'/'+'prediction 2.png'
)

batch_pred_images = pred[2][50:55]
imshow(
    prediction=batch_pred_images, prediction_name='prediction 3', save_name=img_dir+'/'+'prediction 3.png'
)

# %%
batch_raw_images = raw_data[55:60]
imshow(
    raw=batch_raw_images, save_name=img_dir+'/'+'raw.png'
)
# %%
