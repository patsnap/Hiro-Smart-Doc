import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from ..common.stage_timing import StageRecorder
from . import infer_utils
from .backends.base import Backend, RunnerBackend


class BaseRunner:
    def __init__(self, backend: Backend, model: Path, threads: int) -> None:
        self.logger = logging.getLogger("model_runner")
        self.logger.info(f"Using {backend=}")

        self.backend: RunnerBackend

        match backend:
            case Backend.ONNX:
                from .backends.onnx_backend import OnnxBackend

                self.backend = OnnxBackend(model, threads)

            case _:
                raise ValueError("Unknown backend!")


class LayoutModelRunner(BaseRunner):
    def __init__(self, backend: Backend, model: Path, threads: int, input_size: int = 640
        ,layout_classes: list[tuple[str, tuple[int, int, int]]] = None
        ,num_classes: int = 25
        ,complex_class: list[int] = [11, 19, 20, 21, 22, 23],
        conf_thres: float = 0.3) -> None:

        # 首先调用父类构造函数来初始化backend
        super().__init__(backend, model, threads)
        
        self.input_size = input_size
        self.layout_classes = layout_classes
        self.num_classes = num_classes
        self.complex_class = complex_class
        self.duplicate_box_filter = infer_utils.DuplicatesBoxFilter(num_classes=self.num_classes, complex_class=self.complex_class)
        self.conf_thres = conf_thres
    async def inference(self, image: MatLike) -> list[list[float]]:
        # image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        padding_results = self.resize_with_padding(image, (self.input_size, self.input_size), padding_value=114)

        return await self.inference_no_resize(padding_results)

    async def inference_no_resize(self, padding_results: tuple) -> list[list[float]]:
        image = padding_results[0]
        crop_info = padding_results[1]
        # HWC -> BHWC
        _tensor = np.expand_dims(image, axis=0)
        # BGR -> RGB, BHWC -> BCHW, (n, h, w, 3) -> (n, 3, h, w)
        _tensor = _tensor[..., ::-1].transpose((0, 3, 1, 2))
        tensor = np.ascontiguousarray(_tensor.astype(np.float32) / 255.0)

        pred = await self.backend._inference(tensor)
        pred = self.postprocess(pred.reshape(-1, 300, self.num_classes + 4), [image], [crop_info])[
            0
        ]  # batch_size=1

        self.logger.info(f"Layout pred={np.array_repr(pred, precision=5)}")
        return pred.tolist()  # type: ignore

    async def batch_inference(
        self,
        images: list[MatLike],
        *,
        latency: StageRecorder | None = None,
    ) -> list[list[list[float]]]:
        """Run layout detection on multiple images in one forward pass (batched tensor)."""
        if not images:
            return []
        padding_results = [
            self.resize_with_padding(im, (self.input_size, self.input_size), padding_value=114)
            for im in images
        ]
        if latency:
            latency.mark("batch_resize_with_padding")
        padded = [p[0] for p in padding_results]
        crop_infos = [p[1] for p in padding_results]
        _tensor = np.stack(padded, axis=0)
        _tensor = _tensor[..., ::-1].transpose((0, 3, 1, 2))
        tensor = np.ascontiguousarray(_tensor.astype(np.float32) / 255.0)
        if latency:
            latency.mark("batch_tensor_prepare")

        pred = await self.backend._inference(tensor)
        if latency:
            latency.mark("batch_backend_inference")
        batch_size = len(images)
        preds = pred.reshape(batch_size, 300, self.num_classes + 4)
        per_image = self.postprocess(preds, padded, crop_infos)
        if latency:
            latency.mark("batch_postprocess")
        out = [p.tolist() for p in per_image]  # type: ignore
        if latency:
            latency.mark("batch_result_tolist")
        return out

    def postprocess(
        self, preds: NDArray[np.float32], images: list[MatLike], crop_infos: list
    ) -> list[NDArray[np.float32]]:
        # (b, 300, 4+c) -> (b, 300, 4), (b, 300, c)
        bboxes_batch, scores_batch = np.split(preds, (4,), axis=-1)

        results = []
        for i, (bboxes, scores) in enumerate(zip(bboxes_batch, scores_batch)):  # (300, 4), (300, c)
            image = images[i]
            crop_info = crop_infos[i]
            bboxes = infer_utils.xywh2xyxy(bboxes).clip(0, 1)
            bboxes = np.array([self.map_single_box(box, crop_info) for box in bboxes])

            cls = scores.argmax(axis=-1, keepdims=True)  # (300, 1)
            scores = np.take_along_axis(scores, cls, axis=1)  # (300, 1)
            idx = scores.squeeze(-1) > self.conf_thres  # (300, )

            pred = np.concatenate([bboxes, scores, cls], axis=-1)[idx]  # filter: (x, 6)
            pred[:, :4] *= np.concatenate(
                (image.shape[:2][::-1], image.shape[:2][::-1]), axis=-1
            )
            pred = self.duplicate_box_filter.filter(pred)
            pred[:, :4] /= np.concatenate(
                (image.shape[:2][::-1], image.shape[:2][::-1]), axis=-1
            )
            results.append(pred)

        return results

    def duplicate_box_filter(self, pred: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.duplicate_box_filter.filter(pred)

    @staticmethod
    def resize_with_padding(
        img: np.ndarray,
        image_shape: tuple[int, int] | int,
        padding_value: int = 255,
        keep_aspect_ratio: bool = True,
        center_pad: bool = True,
        no_scale_up: bool = True,
        return_crop_info: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, tuple]:
        """
        调整图像大小并添加填充

        Args:
            img: 输入图像 (H,W,C)
            image_shape: 目标尺寸 (height, width) 或单个整数（正方形）
            padding_value: 填充值，默认255（白色）
            keep_aspect_ratio: 是否保持宽高比
            center_pad: 是否居中填充
            no_scale_up: 是否禁止放大
            return_crop_info: 是否返回裁剪信息

        Returns:
            np.ndarray | tuple[np.ndarray, tuple]: 处理后的图像，可选返回裁剪信息
        """
        assert img.ndim == 3
        if isinstance(image_shape, int):
            image_shape = (image_shape, image_shape)

        tgt_height, tgt_width = image_shape[:2]
        ori_h, ori_w = img.shape[:2]

        # 处理不保持宽高比的情况
        if not keep_aspect_ratio:
            img = cv2.resize(img, image_shape[::-1], interpolation=cv2.INTER_LINEAR)
            new_h, new_w = img.shape[:2]
            r_h = new_h / ori_h
            r_w = new_w / ori_w
            if not return_crop_info:
                return img
            else:
                start_x, start_y = 0, 0
                return img, (
                    start_y,
                    start_y + new_h,
                    start_x,
                    start_x + new_w,
                    r_h,
                    r_w,
                )

        # 计算缩放比例
        r = min(tgt_height / ori_h, tgt_width / ori_w)
        if no_scale_up:
            r = min(r, 1.0)

        # 计算新的尺寸
        new_h, new_w = max(math.floor(r * ori_h), 1), max(math.floor(r * ori_w), 1)

        # 调整图像大小
        if (new_h, new_w) == (ori_h, ori_w):
            new_img = img
        else:
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建填充背景
        delta_h = tgt_height - new_h
        delta_w = tgt_width - new_w

        if center_pad:
            start_x = math.floor(delta_w / 2)
            start_y = math.floor(delta_h / 2)
        else:
            start_x, start_y = 0, 0

        # 创建填充背景并放置调整大小后的图像
        bg = np.full(
            (tgt_height, tgt_width, img.shape[2]), padding_value, dtype=img.dtype
        )
        bg[start_y : start_y + new_h, start_x : start_x + new_w] = new_img

        if not return_crop_info:
            return bg
        else:
            # import matplotlib.pyplot as plt
            # plt.imsave("output.png", bg)
            return bg, (start_y, new_h, start_x, new_w)

    def map_single_box(self, box, crop_info):
        # print("box, crop_info", box, crop_info)
        x1, y1, x2, y2 = np.array(box) * self.input_size
        start_y, h, start_x, w = crop_info
        # print("x1, y1, x2, y2", x1, y1, x2, y2)

        # 移除填充
        x1 = max(0, x1 - start_x)
        y1 = max(0, y1 - start_y)
        x2 = max(0, x2 - start_x)
        y2 = max(0, y2 - start_y)

        # 反向缩放
        x1 = x1 / w
        y1 = y1 / h
        x2 = x2 / w
        y2 = y2 / h
        # make sure x1, y1, x2, y2 are in the range of 0-1
        x1 = max(0, min(x1, 1))
        y1 = max(0, min(y1, 1))
        x2 = max(0, min(x2, 1))
        y2 = max(0, min(y2, 1))

        return [x1, y1, x2, y2]

    def draw(self, image: MatLike, bboxes: list[list[float]]) -> MatLike:
        if not bboxes:
            return image

        shape = image.shape[:2]
        scaled_bbox = np.asarray(bboxes, dtype=np.float32)
        scaled_bbox[:, [0, 2]] = (scaled_bbox[:, [0, 2]] * (shape[1] - 1)).round()
        scaled_bbox[:, [1, 3]] = (scaled_bbox[:, [1, 3]] * (shape[0] - 1)).round()

        image_vis = image.copy()
        for x1, y1, x2, y2, score, cls in scaled_bbox:
            cls_str = self.layout_classes[int(cls)][0]
            color = self.layout_classes[int(cls)][1]
            score_str = str(round(score, 2))

            image_vis = cv2.rectangle(
                image_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA
            )
            image_vis = cv2.putText(
                image_vis,
                f"{cls_str} {score_str}",
                (int(x1), int(y1) - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            image = cv2.addWeighted(image_vis, 0.8, image, 0.2, 0)

        return image