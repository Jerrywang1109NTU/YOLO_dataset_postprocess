import numpy as np
import torch
import torchvision

# ---------- 配置参数 ----------
CONF_THRESH = 0.25  # 置信度阈值
IOU_THRESH = 0.45   # NMS 的 IoU 阈值
NUM_CLASSES = 2     # 根据你前面提供的 shape 推测出来的

# ---------- 加载和处理输出 ----------
def reshape_output(output):
    b, h, w, c = output.shape
    assert b == 1
    output = output.reshape(h * w * 3, 7)  # 每个 anchor 7个数：[x, y, w, h, obj_conf, cls1, cls2...]
    return output

def decode_predictions(tensors):
    reshaped = [reshape_output(t) for t in tensors]
    preds = np.concatenate(reshaped, axis=0)  # shape: (N, 7)

    # 置信度乘类别概率，得到最终得分
    obj_conf = preds[:, 4:5]
    cls_scores = preds[:, 5:]  # shape: (N, num_classes)
    confs = obj_conf * cls_scores

    boxes = preds[:, 0:4]  # xywh

    # 转为xyxy格式
    boxes_xyxy = xywh2xyxy(boxes)

    outputs = []
    for cls in range(NUM_CLASSES):
        cls_conf = confs[:, cls]
        mask = cls_conf > CONF_THRESH
        if mask.sum() == 0:
            continue
        cls_boxes = boxes_xyxy[mask]
        scores = cls_conf[mask]

        # NMS
        keep = torchvision.ops.nms(torch.tensor(cls_boxes), torch.tensor(scores), iou_threshold=IOU_THRESH)
        final_boxes = cls_boxes[keep.numpy()]
        final_scores = scores[keep.numpy()]
        final_labels = np.full_like(final_scores, fill_value=cls)

        outputs.append((final_boxes, final_scores, final_labels))

    return outputs

def xywh2xyxy(boxes):
    # 输入: [x_center, y_center, w, h]
    # 输出: [x1, y1, x2, y2]
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy

# ---------- 主程序 ----------
if __name__ == "__main__":
    npz = np.load("dpu_outputs.npz")
    out0, out1, out2 = npz["out0"], npz["out1"], npz["out2"]

    detections = decode_predictions([out0, out1, out2])

    for i, (boxes, scores, labels) in enumerate(detections):
        print(f"Class {i} detections:")