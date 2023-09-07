# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
from tqdm.auto import tqdm
import cv2
import os.path as osp
import os
import pandas as pd
import numpy as np
from PIL import Image
import pdb

def test_single_image(model, img_name, out_dir, color_palette, opacity):
    result = inference_segmentor(model, img_name)
    
    # show the results
    if hasattr(model, 'module'):
        model = model.module

    pallete = None

    img = model.show_result(img_name, result,
                            palette=pallete,
                            show=False, opacity=opacity)

    # save the results
    mmcv.mkdir_or_exist(out_dir)
    out_path = osp.join(out_dir, osp.basename(img_name))
    cv2.imwrite(out_path, img)
    print(f"Result is save at {out_path}")

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file or a directory contains images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff'],
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # from code import InteractiveConsole
    # InteractiveConsole(locals=locals()).interact()
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
    if getattr(model, "PALETTE", None) == None:
        model.PALETTE = None
        
    # check arg.img is directory of a single image.
    df = pd.read_csv("./data/sample_submission.csv")
    result = []
    if osp.isdir(args.img):
        for img in tqdm(os.listdir(args.img)):
            # test_single_image(model, osp.join(args.img, img), args.out, get_palette(args.palette), args.opacity)
            pred = inference_segmentor(model, osp.join(args.img, img))[0]
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred) # 이미지로 변환
            pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            pred = np.array(pred) # 다시 수치로 변환
            # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            for class_id in range(12):
                class_mask = (pred == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)

        df['mask_rle'] = result
        df.to_csv('./baseline_submit.csv', index=False)
    else:
        test_single_image(model, args.img, args.out, get_palette(args.palette), args.opacity)

if __name__ == '__main__':
    main()