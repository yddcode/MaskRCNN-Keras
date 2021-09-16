# -*- coding: utf-8 -*-
"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""
# -*- coding: utf-8 -*-
import os
import sys
import json
import datetime
import numpy as np
import yaml
import cv2
from PIL import Image
import skimage.draw
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project E:/PyCharm2019.2.1    /home/yangdoudou/Mask_RCNN/mrcnn/
ROOT_DIR = '/home/yangdoudou/Mask_RCNN'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import Config
import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class Naxi_chineseConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "naxi"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + naxi + chinese

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

# class Naxi_chineseDataset(utils.Dataset):

#     def load_naxi_chinese(self, dataset_dir, subset):
#         """Load a subset of the Balloon dataset.
#         dataset_dir: Root directory of the dataset.
#         subset: Subset to load: train or val
#         """
#         # Add classes. We have only one class to add.
#         self.add_class("naxi_chinese", 1, "naxi")
#         self.add_class("naxi_chinese", 2, "chinese")

#         # Train or validation dataset?
#         assert subset in ["train", "val"]
#         dataset_dir = os.path.join(dataset_dir, subset)

#         # Load annotations
#         # VGG Image Annotator (up to version 1.6) saves each image in the form:
#         # { 'filename': '28503151_5b5b7ec140_b.jpg',
#         #   'regions': {
#         #       '0': {
#         #           'region_attributes': {},
#         #           'shape_attributes': {
#         #               'all_points_x': [...],
#         #               'all_points_y': [...],
#         #               'name': 'polygon'}},
#         #       ... more regions ...
#         #   },
#         #   'size': 100202
#         # }
#         # We mostly care about the x and y coordinates of each region
#         # Note: In VIA 2.0, regions was changed from a dict to a list.
#         annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
#         annotations = list(annotations.values())  # don't need the dict keys

#         # The VIA tool saves images in the JSON even if they don't have any
#         # annotations. Skip unannotated images.
#         annotations = [a for a in annotations if a['regions']]

#         # Add images
#         for a in annotations:
#             # Get the x, y coordinaets of points of the polygons that make up
#             # the outline of each object instance. These are stores in the
#             # shape_attributes (see json format above)
#             # The if condition is needed to support VIA versions 1.x and 2.x.
#         ####
#             # if type(a['regions']) is dict:
#             #     polygons = [r['shape_attributes'] for r in a['regions'].values()]
#             # else:
#             #     polygons = [r['shape_attributes'] for r in a['regions']] 

#             rects = [r['shape_attributes'] for r in a['regions']]
#             name = [r['region_attributes']['name'] for r in a['regions']]
#             name_dict = {"naxi":1, "chinese":2}
#             name_id = [name_dict[a] for a in name]

#             # load_mask() needs the image size to convert polygons to masks.
#             # Unfortunately, VIA doesn't include it in JSON, so we must read
#             # the image. This is only managable since the dataset is tiny.
#             image_path = os.path.join(dataset_dir, a['filename'])
#             image = skimage.io.imread(image_path)
#             height, width = image.shape[:2]

#             self.add_image(
#                 "naxi_chinese",
#                 image_id=a['filename'],  # use file name as a unique image id
#                 path=image_path,
#                 class_id=name_id,
#                 width=width, height=height,
#                 polygons=rects)

#     def load_mask(self, image_id):
#         """Generate instance masks for an image.
#        Returns:
#         masks: A bool array of shape [height, width, instance count] with
#             one mask per instance.
#         class_ids: a 1D array of class IDs of the instance masks.
#         """
#         # If not a balloon dataset image, delegate to parent class.
#         image_info = self.image_info[image_id]
#         if image_info["source"] != "naxi_chinese":
#             return super(self.__class__, self).load_mask(image_id)
#         print('image_id:', image_info)
#         # Convert polygons to a bitmap mask of shape
#         # [height, width, instance_count]
#         info = self.image_info[image_id]
#         mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
#                         dtype=np.uint8)
#         for i, p in enumerate(info["polygons"]):
#             # Get indexes of pixels inside the polygon and set them to 1
#             rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
#             mask[rr, cc, i] = 1

#         # Return mask, and array of class IDs of each instance. Since we have
#         # one class ID only, we return an array of 1s
#         return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

#     def image_reference(self, image_id):
#         """Return the path of the image."""
#         info = self.image_info[image_id]
#         if info["source"] == "naxi_chinese":
#             return info["path"]
#         else:
#             super(self.__class__, self).image_reference(image_id)

class Naxi_chineseDataset(utils.Dataset):
    
    
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    ## 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            # temp = yaml.load(f.read())
            # print('temp', temp)
            # labels = temp['label_names']
            # print('labels:', labels)
            # del labels[0]
            #with open('odom.txt', 'r') as f:
            labels = f.readlines()
            print('labels:',labels)
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类�?
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        # self.add_class("shapes", 1, "tongue")  # 黑色素瘤
        self.add_class("naxi_chinese", 1, "naxi")
        self.add_class("naxi_chinese", 2, "chinese")
        for i in range(count):
            # 获取图片宽和�?

            filestr = imglist[i].split(".")[0]
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + "_json_label.png"
            # yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/info.yaml"
            yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/label_names.txt"
            # print(dataset_root_path + "/labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "/labelme_json/" + filestr + "_json/" + filestr + "_json_img.png")
            self.add_image("naxi_chinese", image_id=i, path=img_floder + "/" + imglist[i],\
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

            
    # 重写load_mask
    # def load_mask(self, image_id):
    #     """Generate instance masks for shapes of the given image ID.
    #     """
    #     global iter_num
    #     print("image_id", image_id)
    #     info = self.image_info[image_id]
    #     count = 1  # number of object
    #     img = Image.open(info['mask_path'])
    #     num_obj = self.get_obj_index(img)
    #     mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
    #     mask = self.draw_mask(num_obj, mask, img, image_id)
    #     occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    #     for i in range(count - 2, -1, -1):
    #         mask[:, :, i] = mask[:, :, i] * occlusion
    #         occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
    #     labels = []
    #     labels = self.from_yaml_get_class(image_id)
    #     labels_form = []
    #     for i in range(len(labels)):
    #         if labels[i].find("tongue") != -1:
    #             # print "box"
    #             labels_form.append("tongue")
    #     class_ids = np.array([self.class_names.index(s) for s in labels_form])
    #     return mask, class_ids.astype(np.int32)
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id",image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("naxi") != -1:
                # print "box"
                labels_form.append("naxi")
            elif labels[i].find("chinese") != -1:
                #print "column"
                labels_form.append("chinese")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def train_model():
    # 基础设置
    dataset_root_path = r"/home/yangdoudou/Mask_RCNN/dataset"
    img_floder = os.path.join(dataset_root_path, "data")
    mask_floder = os.path.join(dataset_root_path, "mask_png")
    # yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder)
    print('imglist', imglist)
    count = len(imglist)

    # train与val数据集准�?
    dataset_train = Naxi_chineseDataset()
    dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_train.prepare()

    print("dataset_train-->",dataset_train._image_ids)

    dataset_val = Naxi_chineseDataset()
    dataset_val.load_shapes(7, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_val.prepare()   

    # Create models in training mode
    config = Naxi_chineseConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Which weights to start with?
    # 第一次训练时，这里填coco，在产生训练后的模型后，改成last
    init_with = "last"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last models you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                layers="all")


# def train(model):
#     """Train the model."""
#     # Training dataset.
#     dataset_train = Naxi_chineseDataset()
#     dataset_train.load_naxi_chinese('./dataset/data', "train")
#     dataset_train.prepare()

#     # Validation dataset
#     dataset_val = Naxi_chineseDataset()
#     dataset_val.load_naxi_chinese('./dataset/data', "val")
#     dataset_val.prepare()

#     # *** This training schedule is an example. Update to your needs ***
#     # Since we're using a very small dataset, and starting from
#     # COCO trained weights, we don't need to train too long. Also,
#     # no need to train all layers, just the heads should do it.
#     print("Training network heads")
#     model.train(dataset_train, dataset_val,
#                 learning_rate=config.LEARNING_RATE,
#                 epochs=3,
#                 layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


class TongueConfig(Naxi_chineseConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def predict():
    import skimage.io
    import visualize

    # Create models in training mode
    config = Naxi_chineseConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    class_names = ['BG', 'naxi', 'chinese']

    # Load a random image from the images folder
    file_names = r'/home/yangdoudou/Mask_RCNN/images/02.jpg' # next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(file_names)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    print('rreults:', r)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

if __name__ == "__main__":
    #train_model()
    predict()

############################################################
#  Training
############################################################

# if __name__ == '__main__':
#     import argparse

    # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN to detect naxi_chinese.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'splash'")
    # parser.add_argument('--dataset', required=False,
    #                     metavar="/path/to/balloon/dataset/",
    #                     help='Directory of the Balloon dataset')
    # parser.add_argument('--weights', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    # args = parser.parse_args()

    # Validate arguments
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "splash":
    #     assert args.image or args.video,\
    #            "Provide --image or --video to apply color splash"

    # print("Weights: ", args.weights)
    # print("Dataset: ", args.dataset)
    # print("Logs: ", args.logs)

    # Configurations
    # if args.command == "train":
    # config = Naxi_chineseConfig()
    # else:
    #     class InferenceConfig(Naxi_chineseConfig):
    #         # Set batch size to 1 since we'll be running inference on
    #         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #         GPU_COUNT = 1
    #         IMAGES_PER_GPU = 1
    #     config = InferenceConfig()
    # config.display()

    # Create model
    # if args.command == "train":
    # model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir='./logs')
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
                                #   model_dir=args.logs)

    # Select weights file to load
    # if args.weights.lower() == "coco":
    # weights_path = COCO_WEIGHTS_PATH
        # Download weights file
    #     if not os.path.exists(weights_path):
    #         utils.download_trained_weights(weights_path)
    # elif args.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = model.find_last()
    # elif args.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = model.get_imagenet_weights()
    # else:
    #     weights_path = args.weights

    # Load weights
    # print("Loading weights ", weights_path)
    # if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
    # model.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])
    # else:
    #     model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    # if args.command == "train":
    # train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'splash'".format(args.command))
