import torch
from models.common import DetectMultiBackend
from utils.general import cv2, check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np

import cv2


def yolo():
    src = ['input_images/image11.jpg',
           "input_images/forest.jpg",
           "input_images/mavic_image5.jpg",
           'input_images/db5.png',
           "input_images/ap1.jpg"]

    weights = ['weights/yolov5m.pt', 'weights/Bird_drone.pt']

    classes = ['Drone', 'bird']

    frame = cv2.imread(src[-1])

    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 5
    agnostic_nms = False
    imgsz = [640, 640]
    score_th = 0.1

    device = select_device('cpu')
    #weights = r'weights/Drone_weights.pt'
    # weights = r'weights/Bird_drone.pt'
    model = DetectMultiBackend(weights[0], device=device, dnn=False, data=None, fp16=False)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    Iexp = frame

    iexp = letterbox(Iexp, 640, stride=32, auto=True)[0]
    iexp = iexp.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    iexp = np.ascontiguousarray(iexp)
    iexp_tr = torch.from_numpy(iexp).to(device)
    iexp_tr = iexp_tr.half() if model.fp16 else iexp_tr.float()  # uint8 to fp16/32
    iexp_tr /= 255

    if len(iexp_tr.shape) == 3:
        iexp_tr = iexp_tr[None]

    pred = model(iexp_tr)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)

    if len(pred) == 0 or len(pred[0]) == 0:
        pass
        # frame = cv2.putText(frame, "Non drone Image", (50, 100), 0, 2, (0, 0, 255), 2)

    elif len(pred) == 1:
        det = pred[0]
        det[:, :4] = scale_coords(iexp_tr.shape[2:], det[:, :4], Iexp.shape).round()
        det = det.cpu().detach().numpy()

        for predictions in det:

            x1 = int(predictions[0])
            y1 = int(predictions[1])
            w = int(predictions[2] - predictions[0])
            h = int(predictions[3] - predictions[1])

            score = float(predictions[4])
            detection_class = int(predictions[5])
            if score > score_th:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 4)
                frame = cv2.putText(frame, str(np.round(score, 2)), (x1, y1), 0, 0.5, (0, 0, 255), 2)
                if detection_class < 2:
                    frame = cv2.putText(frame, classes[detection_class], (x1 + 50, y1-5), 0, 0.5, (0, 0, 255), 1)
                else:
                    frame = cv2.putText(frame, 'Aeroplane', (x1 + 50, y1-5), 0, 0.5, (0, 0, 255), 1)

    cv2.imwrite("result.jpg", frame)


yolo()
