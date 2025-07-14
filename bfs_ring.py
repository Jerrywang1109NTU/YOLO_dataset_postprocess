import cv2
import numpy as np
from collections import deque
import math
# import matplotlib.pyplot as plt # Removed for PYNQ compatibility
import time

def visualize_results(image, successful_paths, broken_paths, anchor_points, break_points):
    """
    可视化追踪结果并保存到文件。
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

    # 将最终图像保存到文件，而不是显示它
    output_filename = "8_bfs.png"
    cv2.imwrite(output_filename, output_image)
    print(f"结果已保存到文件: {output_filename}")


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
    
    print(f"筛选出 {len(anchors)} 个锚点。")
    # 返回锚点列表、用于追踪的阈值、以及所用的半径
    return anchors, threshold, fixed_radius

def get_ring_search_points(center, min_radius, max_radius):
    """
    获取一个环形区域内的所有像素坐标。
    """
    points = []
    cx, cy = center
    # 遍历一个正方形边界框，然后通过距离检查来筛选出环形区域内的点
    for x in range(cx - max_radius, cx + max_radius + 1):
        for y in range(cy - max_radius, cy + max_radius + 1):
            dist_sq = (x - cx)**2 + (y - cy)**2
            if min_radius**2 <= dist_sq <= max_radius**2:
                points.append((x, y))
    return points

def get_line_pixels(p1, p2):
    """
    使用Bresenham算法获取两点之间直线上的所有像素坐标。
    """
    x0, y0 = p1
    x1, y1 = p2
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def trace_wire_bfs(image, start_point, center, die_rect, visited, threshold, start_r):
    """
    从单个锚点开始，使用一种基于优先级的贪心搜索来追踪导线。
    - image: 灰度图。
    - start_point: 起始锚点。
    - center: 图像中心。
    - die_rect: 中心Die区域的矩形 (x, y, w, h)。
    - visited: 全局访问过的像素记录数组。
    - threshold: 全局灰度阈值，低于此值的像素才被认为是导线的一部分。
    - start_r: 起始环的半径。
    """
    path = [start_point]
    # 标记路径上的所有点为已访问，以避免在同一追踪中回头
    # 使用一个临时访问集，这样每次追踪都是独立的
    temp_visited = set([start_point])

    max_path_len = start_r * 2
    
    # 搜索参数
    search_radius_max = 8
    search_radius_min = 3 # 避免停留在原地
    max_brightness_increase = 20
    max_angle_deviation = 25.0 # 角度容忍度
    path_brightness_tolerance = 20 # 新增：路径亮度容忍度

    for _ in range(max_path_len):
        current_point = path[-1]
        px, py = current_point

        # 检查是否成功到达中心Die区域
        dx, dy, dw, dh = die_rect
        if dx <= px < dx + dw and dy <= py < dy + dh:
            # 成功追踪，标记全局访问数组
            for p in path:
                visited[p[1], p[0]] = 1
            return path, True, None # 路径, 成功状态, 断点

        candidate_points = get_ring_search_points(current_point, search_radius_min, search_radius_max)
        
        valid_candidates = []
        for nx, ny in candidate_points:
            # --- 硬性约束 ---
            # 1. 在图像范围内
            if not (0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]):
                continue
            # 2. 未被访问过 (在本次追踪中)
            # if (nx, ny) in temp_visited:
            #     continue
            # 3. 全局灰度低于阈值
            if image[ny, nx] >= threshold:
                continue
            # 4. 局部亮度增加检查
            if image[ny, nx] > image[py, px] + max_brightness_increase:
                continue
            
            # 5. 新增：路径清空检查
            path_is_clear = True
            line_pixels = get_line_pixels(current_point, (nx, ny))
            num_0 = 0
            num_1 = 0
            # 只检查中间点，不包括起点和终点
            for (lx, ly) in line_pixels[1:-1]:
                if image[ly, lx] > image[py, px] + path_brightness_tolerance or image[ly, lx] > image[ny, nx] + path_brightness_tolerance:
                    # path_is_clear = False
                    # break
                    num_1 += 1
                num_0 += 1
            if num_0 * 0.6 < num_1:
                path_is_clear = False
            if not path_is_clear:
                continue

            # --- 软性约束 (用于排序) ---
            angle_diff = 0.0
            # 6. 角度约束 (路径长度超过10后生效)
            if len(path) >= 4:
                # 使用一个更稳定的历史方向向量
                p_hist_far = path[-3]
                p_hist_near = path[-1]
                vec_hist = (p_hist_near[0] - p_hist_far[0], p_hist_near[1] - p_hist_far[1])
                vec_next = (nx - p_hist_near[0], ny - p_hist_near[1])
                
                dot_product = vec_hist[0] * vec_next[0] + vec_hist[1] * vec_next[1]
                mag_hist = math.sqrt(vec_hist[0]**2 + vec_hist[1]**2)
                mag_next = math.sqrt(vec_next[0]**2 + vec_next[1]**2)

                if mag_hist > 0 and mag_next > 0:
                    cos_angle = max(-1.0, min(1.0, dot_product / (mag_hist * mag_next)))
                    angle_diff = math.degrees(math.acos(cos_angle))
                    if angle_diff > max_angle_deviation:
                        continue # 角度不符合，放弃该候选点
            
            # 使用绝对灰度值进行排序
            gray_value = image[ny, nx]
            # 计算到中心的距离作为排序依据
            dist_to_center = (nx - center[0])**2 + (ny - center[1])**2

            valid_candidates.append({
                "point": (nx, ny),
                "gray_value": gray_value,
                "angle_diff": angle_diff,
                "line_std": num_1 / num_0,
                "dist_to_center": dist_to_center
            })

        # 如果没有找到有效的候选点，则路径中断
        if not valid_candidates:
            for p in path:
                visited[p[1], p[0]] = 1
            return path, False, current_point

        # 动态排序逻辑
        if len(path) < 15:
            # 初始阶段：优先灰度值(越小越好)，其次是离中心近
            valid_candidates.sort(key=lambda c: (c["dist_to_center"], c["line_std"], c["gray_value"], c["angle_diff"]))
        else:
            # 后期阶段：优先灰度值(越小越好)，其次是角度
            valid_candidates.sort(key=lambda c: (c["gray_value"], c["dist_to_center"], c["angle_diff"], c["line_std"]))
        
        best_next_point = valid_candidates[0]["point"]
        
        # 延伸路径
        path.append(best_next_point)
        temp_visited.add(best_next_point)

    # 如果循环因达到最大长度而结束，也视为中断
    for p in path:
        visited[p[1], p[0]] = 1
    return path, False, path[-1]


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

    # 1. 寻找锚点
    fixed_r = int(width * 0.38) 
    anchors, threshold, start_r = find_anchors(image, center, fixed_radius=fixed_r, anchor_min_dist=20)

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
        if len(path) < 3:
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
    visualize_results(original_image, successful_paths, broken_paths, list(anchors), break_points)

if __name__ == '__main__':
    # 请将 '8_13_b_0.jpg' 替换为您上传的图像文件的路径
    # 如果文件在同一目录下，可以直接使用文件名
    # image_file = '8_13_b_0.png'
    image_file = '8_mod.png'
    main(image_file)
