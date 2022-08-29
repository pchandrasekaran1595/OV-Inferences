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

    return model, input_layer, output_layer, \
           (input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="image", help="Mode: image")
    parser.add_argument("--filename", "-f", type=str, default="Test_3.jfif", help="Image Filename")
    parser.add_argument("--target", "-t", type=str, default="CPU", help="Target Device for Inference")
    args = parser.parse_args()

    assert args.filename in os.listdir(INPUT_PATH), "File not Found"
    assert args.target in ["CPU", "GPU"], "Invalid Target Device"

    model, input_layer, output_layer, (N, C, H, W) = setup(args.target)

    if re.match(r"^image$", args.mode, re.IGNORECASE):
        image = cv2.imread(os.path.join(INPUT_PATH, args.filename), cv2.IMREAD_COLOR)
        image = preprocess(image, W, H)

        embeddings = model(inputs=[image])[output_layer]
    

if __name__ == "__main__":
    sys.exit(main() or 0)
