import os
import re
import sys
import cv2
import json
import platform
import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from openvino.runtime import Core


REPO_PATH: str = f"{os.environ['BUILD_DISK']}/Repos/OV-Inferences"
MODEL_PATH=f"{os.environ['BUILD_DISK']}/OVM/ir/public/"

INPUT_PATH: str = os.path.join(REPO_PATH, "input")
LABEL_PATH: str = os.path.join(REPO_PATH, "labels")
MODEL_NAME: str = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]

ID: int = 0
CAM_WIDTH: int  = 640
CAM_HEIGHT: int = 360 
FPS: int = 30


def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")


def preprocess(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    return np.expand_dims(image, axis=0)


def segmenter_decode(class_index_image: np.ndarray) -> np.ndarray:
    colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r, g, b = np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8)

    for i in range(20):
        indexes = (class_index_image == i)
        r[indexes] = colors[i][0]
        g[indexes] = colors[i][1]
        b[indexes] = colors[i][2]
    return np.stack([r, g, b], axis=2)


def show_image(
    image: np.ndarray, 
    cmap: Optional[str]="gnuplot2", 
    title: Optional[str]=None
    ) -> None:

    plt.figure()
    plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


def setup(target: str) -> tuple:
    ie = Core()
    model = ie.read_model(model=f"{MODEL_PATH}/{MODEL_NAME}/FP16/{MODEL_NAME}.xml")
    model = ie.compile_model(model=model, device_name=target)

    input_layer = next(iter(model.inputs))
    output_layer = next(iter(model.outputs))

    labels = json.load(open(os.path.join(LABEL_PATH, "seg_labels_20.json"), "r"))

    return model, labels, input_layer, output_layer, \
           (input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3])


def infer(
    model, 
    output_layer, 
    image: np.ndarray, 
    w: int, 
    h: int,
    labels: list
) -> tuple:

    result = model(inputs=[image])[output_layer].squeeze()
    label_indexes = np.unique(result)

    label_list: list = []
    for index in label_indexes: 
        if index != 0:
            label_list.append(labels[str(index)].title())
    result = cv2.resize(src=segmenter_decode(result), dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    return result, label_list
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="image", help="Mode: image or video or realtime")
    parser.add_argument("--filename", "-f", type=str, default="Test_1.jpg", help="Image or Video Filename")
    parser.add_argument("--downscale", "-ds", type=float, default=None, help="Downscale factor (Useful for Videos)")
    parser.add_argument("--target", "-t", type=str, default="CPU", help="Target Device for Inference")
    args = parser.parse_args()

    assert args.target in ["CPU", "GPU"], "Invalid Target Device"

    model, labels, input_layer, output_layer, (N, C, H, W) = setup(args.target)

    if re.match(r"^image$", args.mode, re.IGNORECASE):
        assert args.filename in os.listdir(INPUT_PATH), "File not Found"

        image = cv2.imread(os.path.join(INPUT_PATH, args.filename), cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        image = preprocess(image, W, H)
        
        result, label_list = infer(model, output_layer, image, w, h, labels)
        
        show_image(result, title=label_list)        
    
    elif re.match(r"^video$", args.mode, re.IGNORECASE):
        assert args.filename in os.listdir(INPUT_PATH), "File not Found"

        cap = cv2.VideoCapture(os.path.join(INPUT_PATH, args.filename))

        while True:
            ret, frame = cap.read()
            if ret:
                if args.downscale:
                    frame = cv2.resize(
                        src=frame, 
                        dsize=(int(frame.shape[1]/args.downscale), int(frame.shape[0]/args.downscale)), 
                        interpolation=cv2.INTER_AREA
                    )
                h, w, _ = frame.shape
                frame = preprocess(frame, W, H)
                frame, _ = infer(model, output_layer, frame, w, h, labels)
                cv2.imshow("Feed", frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif re.match(r"^realtime$", args.mode, re.IGNORECASE):
        if platform.system() != "Windows":
            cap = cv2.VideoCapture(ID)
        else:
            cap = cv2.VideoCapture(ID, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = preprocess(frame, W, H)
            frame, _ = infer(model, output_layer, frame, CAM_WIDTH, CAM_HEIGHT, labels)
            
            cv2.imshow("Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)