import cv2
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
import time

def visualize_results(image, successful_paths, broken_paths, anchor_points, break_points):
    """
    可视化追踪结果。
    - image: 原始灰度图。
    - successful_paths: 追踪成功的路径列表。
    - broken_paths: 追踪失败（断线）的路径列表。
    - anchor_points: 起始锚点列表。
    - break_points: 断点位置列表。
    """
    # 将灰度图转换为彩色图以便标记
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 标记成功路径 (绿色)
    for path in successful_paths:
        for point in path:
            cv2.circle(output_image, point, 1, (0, 255, 0), -1)

    # 标记断开的路径 (红色)
    for path in broken_paths:
        for point in path:
            cv2.circle(output_image, point, 1, (0, 0, 255), -1)

    # 标记起始锚点 (蓝色圆圈)
    for point in anchor_points:
        cv2.circle(output_image, point, 5, (255, 0, 0), 1)

    # 标记断点 (红色X)
    for point in break_points:
        cv2.drawMarker(output_image, point, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

    # 使用 matplotlib 显示图像，因为cv2.imshow在某些环境中可能无法正常工作
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("red: bright paths  x: anchor points of gray  o: anchor points of bright")
    plt.axis('off')
    plt.show()

def get_square_ring_pixels(center, r):
    """
    获取一个正方形环上的所有像素坐标。
    - center: 中心点 (cx, cy)。
    - r: 半径 (环到中心的距离)。
    """
    cx, cy = center
    pixels = []
    # 上下边
    for x in range(cx - r, cx + r + 1):
        pixels.append((x, cy - r))
        pixels.append((x, cy + r))
    # 左右边
    for y in range(cy - r + 1, cy + r):
        pixels.append((cx - r, y))
        pixels.append((cx + r, y))
    return pixels

def judge_edge(x, y, C):
    tmp = 220
    if x <= tmp and y <= tmp:
        return 0
    if x <= tmp and y >= C - tmp:
        return 0
    if x >= C - tmp and y <= tmp:
        return 0
    if x >= C - tmp and y >= C - tmp:
        return 0
    return 1

def judge_anchor(x, y, minx, maxx, miny, maxy): 
    if x == minx or x == maxx or y == miny or y == maxy:
        return 1
    return 0

def find_anchors(image, center, fixed_radius=130, anchor_min_dist=20):
    """
    在固定半径的环上寻找导线的起始锚点。
    1. 在固定半径的环上，计算背景的平均灰度。
    2. 找到环上比背景灰度明显暗的像素作为候选锚点。
    3. 筛选锚点，确保它们之间的像素距离大于指定值。
    """
    height, width = image.shape
    print(fixed_radius)
    fixed_radius = 474 - 20
    # 1. 获取固定半径环上的像素并计算平均灰度
    pixels_coords = get_square_ring_pixels(center, fixed_radius)
    valid_pixels_values = [image[y, x] for x, y in pixels_coords if 0 <= x < width and 0 <= y < height]
    
    if not valid_pixels_values:
        print(f"错误：在半径 {fixed_radius} 处未找到有效像素。")
        return [], 0, 0

    # 将环的平均灰度作为背景参考
    background_mean = np.mean(valid_pixels_values)
    threshold = background_mean - 15 # 使用该环的平均灰度作为追踪的阈值

    print(f"使用固定半径 r = {fixed_radius}, 背景平均灰度 = {background_mean:.2f}")

    # 2. 识别候选锚点
    label_path = '8.txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()
    minx = 100
    maxx = 0
    miny = 100
    maxy = 0
    for line in lines:
        parts = line.strip().split()
        x = float(parts[1])
        y = float(parts[2])
        minx = min(x, minx)
        maxx = max(x, maxx)
        miny = min(y, miny)
        maxy = max(y, maxy)
    
    anchors = []
    for line in lines:
        parts = line.strip().split()
        x = float(parts[1])
        y = float(parts[2])
        if judge_anchor(x, y, minx, maxx, miny, maxy):
            anchors.append((round(x * height), round(y * height)))
    anchors = set(anchors)
    # 3. 筛选锚点，确保最小距离

    # output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for point in anchors:
    #     cv2.circle(output_image, point, 5, (255, 0, 0), 1)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # plt.title("导线追踪结果 (绿色: 成功, 红色: 断线)")
    # plt.axis('off')
    # plt.show()
    
    print(f"筛选出 {len(anchors)} 个锚点。")
    # 返回锚点列表、用于追踪的阈值、以及所用的半径
    return anchors, threshold, fixed_radius

def get_allowed_directions(point, center):
    """
    根据点相对于中心的位置，确定允许的搜索方向（总是朝向中心）。
    - point: 当前点 (px, py)。
    - center: 图像中心 (cx, cy)。
    """
    px, py = point
    cx, cy = center
    
    dx_vec = px - cx
    dy_vec = py - cy

    # 主要在上下两侧
    if abs(dy_vec) > abs(dx_vec):
        if dy_vec < 0: # 在上方，应向下搜索
            return [(0, 1), (-1, 1), (1, 1), (1, 0), (-1, 0), (0, 2)] # 左下, 正下, 右下
        else: # 在下方，应向上搜索
            return [(0, -1), (-1, -1), (1, -1), (1, 0), (-1, 0), (0, -2)] # 左上, 正上, 右上
    # 主要在左右两侧
    else:
        if dx_vec < 0: # 在左侧，应向右搜索
            return [(1, 0), (1, -1), (1, 1), (0, 1), (0, -1), (2, 0)] # 右上, 正右, 右下, 右下
        else: # 在右侧，应向左搜索
            return [(-1, 0), (-1, -1), (-1, 1), (0, 1), (0, -1), (-2, 0)] # 左上, 正左, 左下, 左下

def get_dir(point, center):
    """
    根据点相对于中心的位置，确定允许的搜索方向（总是朝向中心）。
    - point: 当前点 (px, py)。
    - center: 图像中心 (cx, cy)。
    """
    px, py = point
    cx, cy = center
    
    dx_vec = px - cx
    dy_vec = py - cy

    # 主要在上下两侧
    if abs(dy_vec) > abs(dx_vec):
        if dy_vec < 0: # 在上方，应向下搜索
            return 0
        else: # 在下方，应向上搜索
            return 1
    # 主要在左右两侧
    else:
        if dx_vec < 0: # 在左侧，应向右搜索
            return 2
        else: # 在右侧，应向左搜索
            return 3

def trace_wire_bfs(image, start_point, center, die_rect, visited, threshold, start_r):
    """
    从单个锚点开始，使用BFS追踪导线。
    - image: 灰度图。
    - start_point: 起始锚点。
    - center: 图像中心。
    - die_rect: 中心Die区域的矩形 (x, y, w, h)。
    - visited: 全局访问过的像素记录数组。
    - threshold: 全局灰度阈值，低于此值的像素才被认为是导线的一部分。
    - start_r: 起始环的半径。
    """
    q = deque([(start_point, [start_point])]) # 队列中存放 (当前点, 到达该点的路径)
    
    # 检查起始点是否已被访问
    if visited[start_point[1], start_point[0]]:
        return [], False, None
    visited[start_point[1], start_point[0]] = 1
    
    max_path_len = start_r * 2 # 一条路径的最大合理长度
    
    # 定义中心Die区域
    dx, dy, dw, dh = die_rect

    # 定义局部对比度检查的容忍度。
    max_brightness_increase = 20
    # 路径的下一个方向与当前方向的最大允许夹角（度）。
    max_angle_deviation = [0, 50, 20, 10]

    while q:
        (px, py), path = q.popleft()

        # 检查是否成功到达中心Die区域
        if dx <= px < dx + dw and dy <= py < dy + dh:
            return path, True, None # 路径, 成功状态, 断点位置

        # 检查路径是否过长
        if len(path) > max_path_len:
            continue
        
        max_dis = 420
        if get_dir((px, py), center) == 0 and center[1] - py < max_dis: # 上方
            continue
        if get_dir((px, py), center) == 1 and py - center[1] < max_dis: # 下方
            continue
        if get_dir((px, py), center) == 2 and center[0] - px < max_dis: # 左侧
            continue
        if get_dir((px, py), center) == 3 and px - center[0] < max_dis: # 右侧
            continue
        
        # --- 修改：根据当前位置确定搜索方向 ---
        allowed_directions = get_allowed_directions((px, py), center)
        
        # 在允许的方向上搜索邻居
        for i, j in allowed_directions:
            nx, ny = px + i, py + j

            # --- 约束条件 ---
            # 1. 在图像范围内
            if not (0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]):
                continue
            # 2. 未被访问过
            if visited[ny, nx]:
                continue
            # 3. 全局灰度低于阈值
            if image[ny, nx] >= threshold:
                continue
            
            # 4. 局部对比度检查
            if image[ny, nx] > image[py, px] + max_brightness_increase:
                continue

            # --- 修改后的角度控制逻辑 ---
            # 仅在路径足够长时(>=10像素)才开始检查角度
            if len(path) >= 30:
                ok = 1
                # 获取近期历史方向向量 (从倒数第5个点到当前点)，这提供了一个更平滑的历史方向
                for p in range(1, 4):
                    p5 = path[-(p*4)]
                    p10 = path[-(p*8)]
                    vec_recent = (p5[0] - p10[0], p5[1] - p10[1])

                    # 获取新方向向量 (从当前点到下一个候选点)
                    vec_next = (nx - p5[0], ny - p5[1])

                    # 计算两个向量的点积
                    dot_product = vec_recent[0] * vec_next[0] + vec_recent[1] * vec_next[1]

                    # 计算两个向量的模
                    mag_recent = math.sqrt(vec_recent[0]**2 + vec_recent[1]**2)
                    mag_next = math.sqrt(vec_next[0]**2 + vec_next[1]**2)

                    # 避免除以零
                    if mag_recent > 0 and mag_next > 0:
                        # 计算夹角的余弦值
                        cos_angle = dot_product / (mag_recent * mag_next)
                        # 钳制到[-1, 1]范围以防浮点误差
                        cos_angle = max(-1.0, min(1.0, cos_angle))
                        # 计算角度（度）
                        angle_deg = math.degrees(math.acos(cos_angle))

                        if angle_deg > max_angle_deviation[p]:
                            ok = 0 # 角度偏转太大，跳过此点
                            break
                if not ok:
                    continue

            # 5. 必须向中心移动
            dist_curr = (px - center[0])**2 + (py - center[1])**2 # 使用平方距离避免开方
            dist_next = (nx - center[0])**2 + (ny - center[1])**2
            # 为距离检查增加一个容忍度，允许路径有轻微的弯曲
            distance_tolerance = 10.0
            if dist_next > dist_curr + distance_tolerance:
                continue
            
            # 如果满足所有条件
            visited[ny, nx] = 1
            new_path = path + [(nx, ny)]
            q.append(((nx, ny), new_path))

    # 如果队列为空，意味着追踪中断
    return path, False, path[-1] # 路径, 失败状态, 断点位置


def main(image_path):
    """主函数"""
    # 加载图像
    start_time = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"错误：无法加载图像 {image_path}")
        return

    # 对比度较低，可以先进行CLAHE（限制对比度的自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    height, width = image.shape
    center = (width // 2, height // 2)

    # 定义中心Die区域的大致范围 (需要根据实际图像微调)
    die_size = width // 3
    die_rect = (center[0] - die_size // 2, center[1] - die_size // 2, die_size, die_size)

    # 1. 寻找锚点 (MODIFIED SECTION)
    # 使用一个固定的半径来寻找锚点，这个值可以根据图像进行调整
    # 对于这张图，半径在宽度35%到40%左右的位置比较合适
    fixed_r = int(width * 0.38) 
    # 锚点之间的最小像素距离
    anchor_dist = 20 
    anchors, threshold, start_r = find_anchors(image, center, fixed_radius=fixed_r, anchor_min_dist=anchor_dist)

    if not anchors:
        print("未能找到任何锚点，程序退出。")
        return

    # 2. 从每个锚点开始追踪
    visited = np.zeros_like(image, dtype=np.uint8)
    successful_paths = []
    broken_paths = []
    break_points = []

    for anchor in anchors:
        if visited[anchor[1], anchor[0]]:
            continue
        
        path, is_successful, break_point = trace_wire_bfs(
            image, anchor, center, die_rect, visited, threshold, start_r
        )
        
        # 追踪距离过短的路径可能是噪声，忽略
        if len(path) < 10:
            continue

        if is_successful:
            successful_paths.append(path)
        else:
            broken_paths.append(path)
            if break_point:
                break_points.append(break_point)

    print(f"追踪完成。成功连接的导线: {len(successful_paths)}, 检测到断线: {len(broken_paths)}")

    # 3. 可视化结果
    # 重新加载原始图像以获得更清晰的背景
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    end_time = time.time()
    print (end_time - start_time)
    visualize_results(original_image, successful_paths, broken_paths, anchors, break_points)

if __name__ == '__main__':
    # 请将 '8_13_b_0.jpg' 替换为您上传的图像文件的路径
    # 如果文件在同一目录下，可以直接使用文件名
    # image_file = '8_13_b_0.png'
    image_file = '8_mod.png'
    main(image_file)

# benchmark for bfs search method
# this method traces one pixel by one