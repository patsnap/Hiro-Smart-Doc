import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray


def load_image(image_bytes: bytes) -> MatLike:
    return cv2.imdecode(
        np.asarray(bytearray(image_bytes), dtype=np.uint8), cv2.IMREAD_COLOR
    )


def save_image(image: MatLike) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def letterbox(
    img: MatLike,
    new_shape: tuple[int, int] = (960, 512),
    color: tuple[int, int, int] = (114, 114, 114),
) -> MatLike:
    """Resize and pad image while meeting stride-multiple constraints.
    For YOLOv5x6, max stride is 64."""
    shape = img.shape[:2]  # current shape [height, width]
    if shape[0] < shape[1]:  # h < w
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        shape = img.shape[:2]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return img


def non_max_suppression(
    prediction: NDArray[np.float32],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> list[NDArray[np.float32]]:
    xc = prediction[..., 4] > conf_thres  # candidates
    output = [np.zeros((0, 5), dtype=np.float32)] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence

        if not x.shape[0]:  # If none remain process next image
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Detections matrix (xyxy, conf) (cls is ignored here)
        conf = x[:, 5:]
        x = np.concatenate((box, conf), 1)[np.squeeze(conf, -1) > conf_thres]

        if not x.shape[0]:  # Check shape
            continue

        i = nms(x, iou_thres)  # NMS
        x = merge_overlap_boxes(x[i])  # Custom merge

        output[xi] = x

    return output


def xywh2xyxy(x: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # assert (
    #     x.shape[-1] == 4
    # ), f"input shape last dimension expected 4 but input shape is {x.shape}"

    y = np.empty_like(x)  # faster than clone/copy

    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y

    return y


def nms(dets: NDArray[np.float32], thresh: float) -> list[int]:
    """https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def merge_overlap_boxes(dets: NDArray[np.float32]) -> NDArray[np.float32]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]  # sort by areas (reversed)

    order_i = 0
    while order_i < order.shape[0] - 1:
        i = order[order_i]  # i is bigger box
        keep = list(range(order_i + 1))

        for order_j, j in enumerate(order[order_i + 1 :]):  # j is smaller box
            # Reject boxes outside
            if x1[j] >= x2[i] or x2[j] <= x1[i] or y1[j] >= y2[i] or y2[j] <= y1[i]:
                keep.append(order_j + order_i + 1)
                continue

            ovr_x1 = np.maximum(x1[i], x1[j])
            ovr_y1 = np.maximum(y1[i], y1[j])
            ovr_x2 = np.minimum(x2[i], x2[j])
            ovr_y2 = np.minimum(y2[i], y2[j])

            ovr_w = np.maximum(0.0, ovr_x2 - ovr_x1 + 1)
            ovr_h = np.maximum(0.0, ovr_y2 - ovr_y1 + 1)
            # Overlap ratio over smaller box
            ovr_ratio = (ovr_w * ovr_h) / areas[j]

            if ovr_ratio > 0.995:  # Just delete
                continue
            elif ovr_ratio > 0.7:  # Merge and delete
                x1[i] = np.minimum(x1[i], x1[j])
                y1[i] = np.minimum(y1[i], y1[j])
                x2[i] = np.maximum(x2[i], x2[j])
                y2[i] = np.maximum(y2[i], y2[j])
                areas[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
                continue
            else:
                keep.append(order_j + order_i + 1)

        order = order[keep]
        order_i += 1

    return dets[order]  # type: ignore


def scale_coords(
    coords: NDArray[np.float32],
    img0_shape: tuple[int, ...],
    img1_shape: tuple[int, int] = (960, 512),
) -> NDArray[np.float32]:
    # Rescale coords (xyxy) from img1_shape to percentage
    # Rotate coords back
    img0_shape = img0_shape[:2]
    if img0_shape[0] < img0_shape[1]:
        raw_coords = np.copy(coords)
        coords[:, 0] = img1_shape[0] - raw_coords[:, 3]
        coords[:, 1] = raw_coords[:, 0]
        coords[:, 2] = img1_shape[0] - raw_coords[:, 1]
        coords[:, 3] = raw_coords[:, 2]
        img1_shape = img1_shape[::-1]

    # calculate from img0_shape
    gain = min(
        img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
    )  # gain = old / new
    pad = (
        (img1_shape[1] - img0_shape[1] * gain) / 2,
        (img1_shape[0] - img0_shape[0] * gain) / 2,
    )  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding

    coords[:, [0, 2]] /= img0_shape[1] * gain - 1  # x percentage
    coords[:, [1, 3]] /= img0_shape[0] * gain - 1  # y percentage

    coords[:, :4] = coords[:, :4].clip(0, 1)
    return coords




class DuplicatesBoxFilter:
    def __init__(self, num_classes: int = 25, complex_class: list[int] = [11, 19, 20, 21, 22, 23]) -> None:
        self.classes = [i for i in range(num_classes)]
        # 11: figno perhaps locate in the box of figure
        # 19: chem perhaps locate in the box of table
        self.complex_class = complex_class
        self.basic_class = [i for i in self.classes if i not in self.complex_class]
        self.iou_thresh = 0.5
        self.merge_ratio = 0.7
        self.delete_ratio = 0.995

    def filter_basic(self, dets: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        nms agnostic class
        merge agnostic class
        """
        idx = self.nms(dets)
        dets = dets[idx]
        # dets = self.merge_overlap_boxes(dets)
        return dets

    def filter_complex(self, dets: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        nms class-specific
        merge class-specific
        """
        dets_list = []
        for aclass in np.unique(dets[:, 5]):
            if dets[dets[:, 5] == aclass].shape[0] >= 1:
                dets_tmp = dets[dets[:, 5] == aclass]
                idx = self.nms(dets_tmp)
                dets_tmp = dets_tmp[idx]
                # dets_tmp = self.merge_overlap_boxes(dets_tmp)
                dets_list.append(dets_tmp)
        if dets_list:

            return np.concatenate(dets_list, axis=0)
        else:
            return np.ones((0, 6), dtype=np.float32)

    def filter(self, dets: NDArray[np.float32]) -> NDArray[np.float32]:
        """split basic and complex, then filter each of them"""
        dets_basic = dets[np.isin(dets[:, 5], self.basic_class)]
        if dets_basic.shape[0] > 1:
            dets_basic = self.filter_basic(dets_basic)
        dets_complex = dets[np.isin(dets[:, 5], self.complex_class)]
        if dets_complex.shape[0] > 1:
            dets_complex = self.filter_complex(dets_complex)
        return np.concatenate([dets_basic, dets_complex], axis=0)

    def merge_overlap_boxes(self, dets: NDArray[np.float32]) -> NDArray[np.float32]:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = areas.argsort()[::-1]  # sort by areas (reversed)

        order_i = 0
        while order_i < order.shape[0] - 1:
            i = order[order_i]  # i is bigger box
            keep = list(range(order_i + 1))

            for order_j, j in enumerate(order[order_i + 1 :]):  # j is smaller box
                # Reject boxes outside
                if x1[j] >= x2[i] or x2[j] <= x1[i] or y1[j] >= y2[i] or y2[j] <= y1[i]:
                    keep.append(order_j + order_i + 1)
                    continue

                ovr_x1 = np.maximum(x1[i], x1[j])
                ovr_y1 = np.maximum(y1[i], y1[j])
                ovr_x2 = np.minimum(x2[i], x2[j])
                ovr_y2 = np.minimum(y2[i], y2[j])

                ovr_w = np.maximum(0.0, ovr_x2 - ovr_x1 + 1)
                ovr_h = np.maximum(0.0, ovr_y2 - ovr_y1 + 1)
                # Overlap ratio over smaller box
                ovr_ratio = (ovr_w * ovr_h) / areas[j]

                if ovr_ratio > self.delete_ratio:  # Just delete
                    continue
                elif ovr_ratio > self.merge_ratio:  # Merge and delete
                    x1[i] = np.minimum(x1[i], x1[j])
                    y1[i] = np.minimum(y1[i], y1[j])
                    x2[i] = np.maximum(x2[i], x2[j])
                    y2[i] = np.maximum(y2[i], y2[j])
                    areas[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
                    continue
                else:
                    keep.append(order_j + order_i + 1)

            order = order[keep]
            order_i += 1

        return dets[order]  # type: ignore

    def nms(self, dets: NDArray[np.float32]) -> list[int]:
        """https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # iou
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # Overlap ratio over smaller box
            ovr_ratio = inter / np.minimum(areas[i], areas[order[1:]])

            inds = np.where((ovr <= self.iou_thresh) & (ovr_ratio <= self.merge_ratio))[
                0
            ]
            order = order[inds + 1]

        return keep
