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

    labels = json.load(open(os.path.join(LABEL_PATH, "coco_labels_80.json"), "r"))

    return model, labels, input_layer, output_layer, \
           (input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3])


def infer_best_box(
    model, 
    output_layer, 
    image: np.ndarray, 
    w: int, 
    h: int) -> tuple:

    result = model(inputs=[image])[output_layer].squeeze()
    
    probs = result[0][2]
    best_index = np.argmax(probs)

    label = int(result[best_index][1])
    x1 = int(result[best_index][3] * w)
    y1 = int(result[best_index][4] * h)
    x2 = int(result[best_index][5] * w)
    y2 = int(result[best_index][6] * h)

    return label, probs, (x1, y1), (x2, y2)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="image", help="Mode: image or video or realtime")
    parser.add_argument("--filename", "-f", type=str, default="Test_2.jpg", help="Image or Video Filename")
    parser.add_argument("--downscale", "-ds", type=float, default=None, help="Downscale factor (Useful for Videos)")
    parser.add_argument("--target", "-t", type=str, default="CPU", help="Target Device for Inference")
    args = parser.parse_args()

    assert args.filename in os.listdir(INPUT_PATH), "File not Found"
    assert args.target in ["CPU", "GPU"], "Invalid Target Device"

    model, labels, input_layer, output_layer, (N, C, H, W) = setup(args.target)

    if re.match(r"^image$", args.mode, re.IGNORECASE):
        image = cv2.imread(os.path.join(INPUT_PATH, args.filename), cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        disp_image = image.copy()
        image = preprocess(image, W, H)

        label_index, probs, (x1, y1), (x2, y2) = infer_best_box(model, output_layer, image, w, h)
        cv2.rectangle(disp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        show_image(disp_image, title=f"{labels[str(label_index)].title()} ({probs:.2f})")

    
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
                disp_frame = frame.copy()
                h, w, _ = disp_frame.shape
                frame = preprocess(frame, W, H)
                label_index, probs, (x1, y1), (x2, y2) = infer_best_box(model, output_layer, frame, w, h)

                cv2.putText(        
                    img=disp_frame, 
                    text=f"{labels[str(label_index)].title()} ({probs:.2f})", 
                    org=(x1-10, y1-10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0, 255, 0), 
                    thickness=1
                )
                cv2.imshow("Feed", disp_frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
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
            disp_frame = frame.copy()
            if not ret: break
            
            frame = preprocess(frame, W, H)
            label_index, probs, (x1, y1), (x2, y2) = infer_best_box(model, output_layer, frame, CAM_WIDTH, CAM_HEIGHT)

            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(        
                img=disp_frame, 
                text=f"{labels[str(label_index)].title()} ({probs:.2f})", 
                org=(x1-10, y1-10), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, 
                color=(0, 255, 0), 
                thickness=1
            )
            cv2.imshow("Feed", disp_frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)