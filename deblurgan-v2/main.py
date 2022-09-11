import os
import re
import sys
import cv2
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


def show_images(
    image_1: np.ndarray,
    image_2: np.ndarray, 
    cmap: Optional[str]="gnuplot2", 
    title: Optional[str]=None
    ) -> None:

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(src=image_1, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(src=image_2, code=cv2.COLOR_BGR2RGB), cmap=cmap)
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
    parser.add_argument("--mode", "-m", type=str, default="image", help="Mode: image or video or realtime")
    parser.add_argument("--filename", "-f", type=str, default="Test_1.jpg", help="Image or Video Filename")
    parser.add_argument("--downscale", "-ds", type=float, default=None, help="Downscale factor (Useful for Videos)")
    parser.add_argument("--target", "-t", type=str, default="CPU", help="Target Device for Inference")
    args = parser.parse_args()

    assert args.filename in os.listdir(INPUT_PATH), "File not Found"
    assert args.target in ["CPU", "GPU"], "Invalid Target Device"

    model, input_layer, output_layer, (N, C, H, W) = setup(args.target)

    if re.match(r"^image$", args.mode, re.IGNORECASE):
        image = cv2.GaussianBlur(src=cv2.imread(os.path.join(INPUT_PATH, args.filename), cv2.IMREAD_COLOR), ksize=(51, 51), sigmaX=5)
        disp_image = image.copy()
        h, w, _ = image.shape
        image = preprocess(image, W, H)

        result = model(inputs=[image])[output_layer].squeeze().transpose(1, 2, 0)
        result = cv2.resize(src=result, dsize=(w, h), interpolation=cv2.INTER_AREA)

        show_images(disp_image, result)
    
    elif re.match(r"^video$", args.mode, re.IGNORECASE):
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
                
                result = model(inputs=[frame])[output_layer].squeeze().transpose(1, 2, 0)
                result = cv2.resize(src=result, dsize=(w, h), interpolation=cv2.INTER_AREA)

                cv2.imshow("Feed", result)
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
            frame = cv2.GaussianBlur(src=frame, ksize=(35, 35), sigmaX=5)
            if not ret: break
            
            h, w, _ = frame.shape
            frame = preprocess(frame, W, H)
            
            result = model(inputs=[frame])[output_layer].squeeze().transpose(1, 2, 0)
            result = cv2.resize(src=result, dsize=(w, h), interpolation=cv2.INTER_AREA)

            cv2.imshow("Feed", result)
        
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)