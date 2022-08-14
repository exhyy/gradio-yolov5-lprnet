import gradio as gr
import numpy as np
import torch
import os

from utils.torch_utils import select_device, time_sync
from detect import load_model as load_model_yolov5
from detect import run as run_yolov5
from utils.augmentations import letterbox
from detect_gardio import predict
from models.LPRNet import build_lprnet
from data.load_data import CHARS

device = select_device('0')
yolov5_model = load_model_yolov5('./weights/yolov5s_best.pt', device=device, data='./data/license.yaml')
lprnet_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
lprnet_model.load_state_dict(torch.load('./weights/lprnet_best.pth', map_location="cpu"))
lprnet_model.to(device)

models = {'yolov5': yolov5_model, 'lprnet': lprnet_model}

def image_classifier(inp):
    # print(type(inp))
    # print(inp.shape)
    # inp1 = letterbox(inp, new_shape=640, stride=32, auto=True)[0]
    # inp1 = inp1.transpose((2, 0, 1))[::-1]
    # inp1 = np.ascontiguousarray(inp1)
    # print(inp1.shape)
    out = predict(inp, models)
    print('------ OK ------')
    # out = cv2.imread("./cropped/cropped.jpg")
    # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    os.remove("./cropped/cropped.jpg")
    return out

if __name__ == '__main__':
    demo = gr.Interface(fn=image_classifier, inputs="image", outputs="image",
        title="Yolov5+LRPNet", description="# Yolov5+LPRNet车牌识别")
    demo.launch()