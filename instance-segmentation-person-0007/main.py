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
MODEL_PATH=f"{os.environ['BUILD_DISK']}/OVM/models/intel/"

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
    image_3: np.ndarray, 
    cmap_1: Optional[str]="gnuplot2",
    cmap_2: Optional[str]="gray", 
    cmap_3: Optional[str]="gnuplot2",
    title_1: Optional[str]="Detection Box",
    title_2: Optional[str]="Mask",
    title_3: Optional[str]="Masked",
    ) -> None:

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(src=image_1, code=cv2.COLOR_BGR2RGB), cmap=cmap_1)
    if title_1: plt.title(title_1)
    plt.subplot(1, 3, 2)
    plt.imshow(image_2, cmap=cmap_2)
    if title_2: plt.title(title_2)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(src=image_3, code=cv2.COLOR_BGR2RGB), cmap=cmap_3)
    if title_3: plt.title(title_3)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


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

    return model, input_layer, model.outputs, \
           (input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3])
    

def infer(
    model,
    output_layer,
    image: np.ndarray,
    w: int, 
    h: int,
    W: int=544,
    H: int=320) -> tuple:
    
    best_box  = model(inputs=[image])[output_layer[0]].squeeze()[0]
    best_mask = cv2.resize(src=model(inputs=[image])[output_layer[2]].squeeze()[0], dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    best_mask[best_mask <= 0] = 0
    best_mask[best_mask > 1] = 1
    best_mask = np.clip(255*best_mask, 0, 255).astype("uint8")

    x1 = int(w * best_box[0] / W)
    y1 = int(h * best_box[1] / H)
    x2 = int(w * best_box[2] / W)
    y2 = int(h * best_box[3] / H)

    return (x1, y1), (x2, y2), (best_mask)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="image", help="Mode: image or video or realtime")
    parser.add_argument("--filename", "-f", type=str, default="Test_11.jpg", help="Image or Video Filename")
    parser.add_argument("--downscale", "-ds", type=float, default=None, help="Downscale factor (Useful for Videos)")
    parser.add_argument("--target", "-t", type=str, default="CPU", help="Target Device for Inference")
    args = parser.parse_args()

    assert args.target in ["CPU", "GPU"], "Invalid Target Device"

    model, input_layer, output_layer, (N, C, H, W) = setup(args.target)

    if re.match(r"^image$", args.mode, re.IGNORECASE):
        assert args.filename in os.listdir(INPUT_PATH), "File not Found"

        image = cv2.imread(os.path.join(INPUT_PATH, args.filename), cv2.IMREAD_COLOR)
        disp_image_1 = image.copy()
        disp_image_2 = image.copy()
        h, w, _ = disp_image_1.shape
        image = preprocess(image, W, H)

        (x1, y1), (x2, y2), best_mask = infer(model, output_layer, image, w, h)
        cv2.rectangle(disp_image_1, (x1, y1), (x2, y2), (0, 255, 0), thickness=int(w/200))
        for i in range(3): disp_image_2[:, :, i] = disp_image_2[:, :, i] & best_mask
        
        show_images(disp_image_1, best_mask, disp_image_2)


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
                disp_frame = frame.copy()
                h, w, _ = disp_frame.shape
                frame = preprocess(frame, W, H)
                (x1, y1), (x2, y2), best_mask = infer(model, output_layer, frame, w, h)
                cv2.rectangle(disp_frame_1, (x1, y1), (x2, y2), (0, 255, 0), thickness=int(w/200))
                for i in range(3): disp_frame_2[:, :, i] = disp_frame_2[:, :, i] & best_mask

                disp_frame = np.concatenate((
                    disp_frame_1,
                    disp_frame_2
                ), axis=1)

                cv2.imshow("Feed", disp_frame)
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
            disp_frame_1 = frame.copy()
            disp_frame_2 = frame.copy()
            h, w, _ = disp_frame_1.shape

            if not ret: break
            
            frame = preprocess(frame, W, H)
            (x1, y1), (x2, y2), best_mask = infer(model, output_layer, frame, w, h)
            cv2.rectangle(disp_frame_1, (x1, y1), (x2, y2), (0, 255, 0), thickness=int(w/200))
            for i in range(3): disp_frame_2[:, :, i] = disp_frame_2[:, :, i] & best_mask

            disp_frame = np.concatenate((
                disp_frame_1,
                disp_frame_2
            ), axis=1)

            cv2.imshow("Feed", disp_frame)
        
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)