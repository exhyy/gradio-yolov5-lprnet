import cv2
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, save_one_box, colors
from models.common import DetectMultiBackend
from pathlib import Path
from data.load_data import CHARS


def predict(img0, models):
    assert isinstance(img0, np.ndarray)
    yolov5_model, lpr_model = models['yolov5'], models['lprnet']
    assert isinstance(yolov5_model, DetectMultiBackend)
    stride, names, pt = yolov5_model.stride, yolov5_model.names, yolov5_model.pt

    img = letterbox(img0, new_shape=640, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(yolov5_model.device)
    # print(yolov5_model.device)
    img = img.half() if yolov5_model.fp16 else img.float()
    img /= 255
    img = img.unsqueeze(0)

    pred = yolov5_model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45,
        classes=None, agnostic=False, max_det=1000)
    
    for i, det in enumerate(pred):
        # img_cropped, img0_ = img.copy(), img0.copy()
        img0_ = img0.copy()
        img_cropped = img0_.copy()
        annotator = Annotator(img0_, line_width=4, example="粤")
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_.shape).round()
            for *xyxy, conf, cls in reversed(det):
                save_one_box(xyxy, img_cropped, file=Path('./cropped/cropped.jpg'), BGR=False)
                label = LPR_predict(lpr_model, './cropped/cropped.jpg', size=[94, 24], device=yolov5_model.device)
                label = f"{label} {conf:.2f}"
                # label = f"{conf:.2f}"
                annotator.box_label(xyxy, label)
    
    img0_ = annotator.result()
    # img0_ = cv2.cvtColor(img0_, cv2.COLOR_BGR2RGB)
    return img0_

    
    return label

def LPR_predict(model, filename: str, size, device):
    img = cv2.imread(filename)
    height, width, _ = img.shape
    if height != size[1] or width != size[0]:
        img = cv2.resize(img, size)
    img = torch.from_numpy(LPR_transform(img))
    img = img.unsqueeze(0)
    img_np = img.numpy().copy()
    img = img.to(device)
    model = model.to(device)

    preds = model(img)
    preds = preds.cpu().detach().numpy()
    pred_labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
    label = ''
    for i in pred_labels[0]:
        label += CHARS[i]
    return label

def LPR_transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))

    return img
    
    