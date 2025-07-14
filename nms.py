import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def draw_detections(image, boxes, scores, class_ids, class_names=None, save_path='result.jpg'):
    """
    将检测结果绘制在图片上
    """
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        # 转成 int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 画矩形
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 标签文字
        label = f"{class_names[class_id] if class_names else 'cls_'+str(class_id)} {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 保存
    cv2.imwrite(save_path, image)
    print(f"Detection result saved to {save_path}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_single_scale(output, stride, conf_threshold=0.5):
    """
    对单层特征图做解码
    output: numpy array, shape=(H, W, 21)
    stride: 当前层对应的 stride
    """
    H, W, C = output.shape
    num_anchors = 3
    num_classes = 2
    anchor_dim = 5 + num_classes

    output = output.reshape(H, W, num_anchors, anchor_dim)

    boxes = []
    scores = []
    class_ids = []

    for y in range(H):
        for x in range(W):
            for anchor in range(num_anchors):
                data = output[y, x, anchor]
                obj_conf = sigmoid(data[4])
                class_scores = sigmoid(data[5:])
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                final_conf = obj_conf * class_conf

                if final_conf < conf_threshold:
                    continue

                bx = (sigmoid(data[0]) + x) * stride
                by = (sigmoid(data[1]) + y) * stride
                bw = np.exp(data[2]) * stride
                bh = np.exp(data[3]) * stride

                x1 = bx - bw / 2
                y1 = by - bh / 2
                x2 = bx + bw / 2
                y2 = by + bh / 2

                boxes.append([x1, y1, x2, y2])
                scores.append(final_conf)
                class_ids.append(class_id)

    return boxes, scores, class_ids

def multi_scale_post_process(dpu_outputs, conf_threshold=0.5, nms_threshold=0.45, img_size=640):
    """
    dpu_outputs: [output0, output1, output2], 每层形状 (H, W, C)
    """
    strides = [8, 16, 32]  # 对应 80x80, 40x40, 20x20 层的 stride
    all_boxes = []
    all_scores = []
    all_class_ids = []

    for i, output in enumerate(dpu_outputs):
        boxes, scores, class_ids = decode_single_scale(output[0], strides[i], conf_threshold)
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_class_ids.extend(class_ids)

    boxes = np.array(all_boxes)
    scores = np.array(all_scores)
    class_ids = np.array(all_class_ids)

    keep = nms(boxes, scores, nms_threshold)

    final_boxes = boxes[keep]
    final_scores = scores[keep]
    final_class_ids = class_ids[keep]

    return final_boxes, final_scores, final_class_ids

def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

if __name__ == "__main__":
    # 假设你已经跑完 multi_scale_post_process 得到以下变量
    start_time = time.time()
    data = np.load('dpu_outputs.npz')
    output0 = data['out0']
    output1 = data['out1']
    output2 = data['out2']

    final_boxes, final_scores, final_class_ids = multi_scale_post_process(
        [output0],
        conf_threshold=0.5,
        nms_threshold=0.1,
        img_size=640
    )
    end_time = time.time()
    print(f"Post-processing time: {end_time - start_time:.4f} seconds")
    # 原图路径（输入给DPU时用的原图路径）
    image_path = '8_13_b_0.png'
    # image_path = '8_mod.png'

    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))

    # 类别名（举例）
    class_names = ['A', 'B']
    
    # 绘图
    draw_detections(image, final_boxes, final_scores, final_class_ids, class_names, save_path='result_single_channel.png')