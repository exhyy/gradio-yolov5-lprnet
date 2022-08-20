import gradio as gr
import tempfile
import torch
import os
from models.LPRNet import build_lprnet
from detect import load_model as load_model_yolov5
from utils.torch_utils import select_device
from data.load_data import CHARS
from detect_gardio import predict

yolov5_model = None
lprnet_model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
device = select_device()
models = None

def upload_weights(file1, file2):
    global yolov5_model, models
    assert isinstance(file1, tempfile._TemporaryFileWrapper)
    try:
        yolov5_model = load_model_yolov5(file1.name, device=device, data='./data/license.yaml')
    except Exception as e:
        print("Exception: ", e)
        return "yolov5模型加载失败！"
    try:
        lprnet_model.load_state_dict(torch.load(file2.name, map_location="cpu"))
    except Exception as e:
        print("Exception: ", e)
        return "LPRNet模型加载失败！"
    lprnet_model.to(device)
    models = {"yolov5": yolov5_model, "lprnet": lprnet_model}
    return "模型加载成功"

def display_image_uploader(upload_info):
    if upload_info.find("成功") != -1:
        return gr.Column.update(visible=True)
    else:
        return gr.Column.update(visible=False)

def image_predict(input_image):
    out = predict(input_image, models)
    os.remove("./cropped/cropped.jpg")
    return out

if __name__ == "__main__":
    demo = gr.Blocks(title="车牌检测")
    with demo:
        with gr.Column():
            gr.Markdown("# <center>基于YOLOv5和LPRNet的车牌检测模型</center>")
            gr.Markdown("## 第一步：上传YOLOv5模型和LPRNet模型")
            gr.Markdown("点击选择YOLOv5模型")
            yolov5_weights_file = gr.File(interactive=True)
            gr.Markdown("点击选择LPRNet模型")
            lpr_weigths_file = gr.File(interactive=True)
            with gr.Row():
                weights_upload_btn = gr.Button("上传模型")
                gr.Variable()
            upload_info_md = gr.Markdown("")
            image_uploader_col = gr.Column(visible=False)
            with image_uploader_col:
                step2_md = gr.Markdown("## 第二步：选择图片，检测车牌信息")
                with gr.Row():
                    input_image = gr.Image(interactive=True)
                    output_image = gr.Image(interactive=False)
                with gr.Row():
                    image_upload_btn = gr.Button("上传图片")
                    gr.Variable()
                
        weights_upload_btn.click(upload_weights, [yolov5_weights_file, lpr_weigths_file], upload_info_md)
        upload_info_md.change(display_image_uploader, upload_info_md, image_uploader_col)
        image_upload_btn.click(image_predict, input_image, output_image)

    demo.launch(debug=True)