import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
from torchtoolbox.transform import Cutout
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA


class ImageFeature:
    def __init__(self, adata, pca_components=50, verbose=False, seeds=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnn_type = 'ResNet50'

    def load_cnn_model(self):
        cnn_pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        cnn_pretrained_model.fc = nn.Identity()
        cnn_pretrained_model = cnn_pretrained_model.to(device=self.device)
        return cnn_pretrained_model

    def extract_image_features(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]),
                          ]

        image_to_tensor = transforms.Compose(transform_list)
        model = self.load_cnn_model()
        model.eval()
        feature_df = pd.DataFrame()
        if 'slice_path' not in self.adata.obs.keys():
            raise ValueError('Please run the function image_crop first')

        with tqdm(total=len(self.adata), desc='Extract image feature',
                  bar_format='{l_bar}{bar} [ time left: {remaining} ]', ) as pbar:
            for spot, slice_path in self.adata.obs['slice_path'].items():
                spot_slice = Image.open(slice_path)
                spot_slice = spot_slice.resize((224, 224))
                spot_slice = np.asarray(spot_slice, dtype=np.float32)
                spot_slice = spot_slice / 255.0

                tensor = image_to_tensor(spot_slice)
                tensor = tensor.reshape(1, 3, 224, 224)
                tensor = tensor.to(self.device)
                result = model(tensor)
                result_npy = result.data.cpu().numpy().ravel()
                feature_df[spot] = result_npy
                feature_df = feature_df.copy()
                pbar.update(1)

        self.adata.obsm['image_feature'] = feature_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feature']")

        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feature_df.transpose().to_numpy())
        self.adata.obsm['image_feat_pca'] = pca.transform(feature_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image features is added to adata.obsm['image_feat_pca']")
        return self.adata


def image_crop(adata, save_path, library_id=None, crop_size=112, target_size=224, verbose=False, full_image_path=None):
    if library_id is None:
        library_id = list(adata.uns['spatial'].keys())[0]
    # load hires quality image
    image = adata.uns['spatial'][library_id]['images']['hires']
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
    image_coord = adata.obsm['spatial'] * scale
    adata.obs['image_col'] = image_coord[:, 0]
    adata.obs['image_row'] = image_coord[:, 1]
    image_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(total=len(adata), desc='Tiling Image', bar_format='{l_bar}{bar} [ time left: {remaining} ]') as pbar:
        for image_row, image_col in zip(adata.obs['image_row'], adata.obs['image_col']):
            image_down = image_row - crop_size / 2
            image_up = image_row + crop_size / 2
            image_left = image_col - crop_size / 2
            image_right = image_col + crop_size / 2

            tile = image_pillow.crop(
                (image_left, image_down, image_right, image_up)
            )
            tile.thumbnail((target_size, target_size), Image.LANCZOS)
            tile.resize((target_size, target_size))
            tile_name = str(image_col) + '-' + str(image_row) + '-' + str(crop_size)
            if save_path is not None:
                out_tile = Path(save_path) / (tile_name + '.png')
                tile_names.append(str(out_tile))
                if verbose:
                    print('Generating tile at location ({}, {})'.format(str(image_col), str(image_row)))
                tile.save(out_tile, 'PNG')
            pbar.update(1)

    adata.obs['slice_path'] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slice_path']")
    return adata
