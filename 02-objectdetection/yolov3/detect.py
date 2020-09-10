# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 2019-11-18
# usage: do object detection
# --------------------

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
import torch
import torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to the definition of model")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights")
    parser.add_argument("--class_path", type=str, default="data/coco.names")
    parser.add_argument("--conf_thres", type=float, default=0.8)
    parser.add_argument("--num_thres", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpu", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--checkpoint_model", type=str)
    opt = parser.parse_args()
    print (opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok = True)

    # set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswidth('.weights'):
        # load model weihghts
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size = opt.batch_size,
        shuffle = False,
        num_workers = opt.n_cpu,
    )

    classes = load_classes(opt.class_path)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []
    img_detections = []

    print("performing object detecting")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))

        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.num_thres)

        current_time = time.time()
        inference_time = datetime.timedelta(seconds= current_time - prev_time)
        prev_time = current_time

        print ("\t +batch %d, inference time: %s"%(batch_i, inference_time))

        imgs.extens(img_paths)
        img_detections.extend(detections)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\n saving images:")
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("Image index is : {}, name is :{}".format(img_i, path))

        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplot(1)
        ax.show(img)

        # draw the detection bbox on the image
        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_pred = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_pred)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t + label: %s, conv:%.5f"%(classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # create a rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(bbox)
                plt.text(
                    x1,
                    y1,
                    s = classes[int(cls_pred)],
                    color='white',
                    verticalalignment='top',
                    bbox={"color": color, "pad":0}
                )
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/").split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches='tight', pad_inches = 0.0)
        plt.close()