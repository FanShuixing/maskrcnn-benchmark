import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import argparse
import cv2
import os
from tqdm import tqdm

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


def load(img_path):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(img_path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument('--config', default='/output/tf_dir/config.yml')
    parser.add_argument('--model_weights', default='/output/tf_dir/model_final.pth')
    parser.add_argument('--input_root', default='/input0/train_0712/images/')
    parser.add_argument('--output_dir', default='./predictions')
    args = parser.parse_args()
    # update the config options with the config file
    cfg.merge_from_file(args.config)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        weight_loading=args.model_weights
    )
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for img_name in tqdm(os.listdir(args.input_root)[:10]):
        img = load(os.path.join(args.input_root, img_name))
        predictions = coco_demo.run_on_opencv_image(img)
        cv2.imwrite('%s/%s' % (args.output_dir, img_name), predictions)


if __name__ == '__main__':
    main()
