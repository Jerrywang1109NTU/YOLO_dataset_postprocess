import cv2
import numpy as np
from collections import deque
import math
# import matplotlib.pyplot as plt # Removed for PYNQ compatibility
import time

def visualize_results(image, successful_paths, broken_paths, anchor_points, second_anchor_points, break_points):
    """
    可视化追踪结果并保存到文件。
    - image: 原始灰度图。
    - successful_paths: 追踪成功的路径列表。
    - broken_paths: 追踪失败（断线）的路径列表。
    - anchor_points: 起始锚点列表。
    - second_anchor_points: 第二层锚点列表。
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
        cv2.circle(output_image, point, 5, (255, 0, 0), 2)

    # 新增：标记第二层锚点 (黄色圆圈)
    for point in second_anchor_points:
        cv2.circle(output_image, point, 5, (0, 255, 255), 1)

    # 标记断点 (红色X)
    for point in break_points:
        cv2.circle(output_image, point, 5, (0, 0, 255), 1)

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
        return [], [], 0, 0

    # 将环的平均灰度作为背景参考
    background_mean = np.mean(valid_pixels_values)
    threshold = background_mean - 15 # 使用该环的平均灰度作为追踪的阈值

    print(f"使用固定半径 r = {fixed_radius}, 背景平均灰度 = {background_mean:.2f}")

    # 2. 识别候选锚点
    label_path = '8.txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    all_points = []
    for line in lines:
        parts = line.strip().split()
        all_points.append((float(parts[1]), float(parts[2])))

    # 找出两层边界
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    x_coords.sort()
    y_coords.sort()
    x_coords = np.unique(x_coords)
    y_coords = np.unique(y_coords)
    
    minxx, minx = x_coords[0], x_coords[1]
    maxxx, maxx = x_coords[-1], x_coords[-2]
    minyy, miny = y_coords[0], y_coords[1]
    maxyy, maxy = y_coords[-1], y_coords[-2]

    outer_anchors_norm = {p for p in all_points if judge_anchor(p[0], p[1], minxx, maxxx, minyy, maxyy)}
    inner_anchors_norm = {p for p in all_points if judge_anchor(p[0], p[1], minx, maxx, miny, maxy)}

    outer_anchors = {(round(x * width), round(y * height)) for x, y in outer_anchors_norm}
    inner_anchors_list = [(round(x * width), round(y * height)) for x, y in inner_anchors_norm]

    anchor_pairs = []
    if not inner_anchors_list:
        print("警告: 未找到第二层锚点。")
        return [], [], threshold, fixed_radius

    for oa in outer_anchors:
        min_dist_sq = float('inf')
        closest_ia = None
        for ia in inner_anchors_list:
            dist_sq = (oa[0] - ia[0])**2 + (oa[1] - ia[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_ia = ia
        if closest_ia:
            anchor_pairs.append((oa, closest_ia))

    print(f"筛选出 {len(anchor_pairs)} 对锚点。")
    return anchor_pairs, threshold, fixed_radius

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

def trace_wire_bfs(image, start_point, second_anchor, center, die_rect, visited, threshold, start_r):
    """
    从单个锚点开始，使用BFS思想来追踪导线。
    """
    # 1. 使用队列来实现BFS
    q = deque([(start_point, [start_point])]) # (current_point, path_to_it)
    
    # 2. 临时访问集合，防止在单次追踪中形成环路
    temp_visited = {start_point, second_anchor}

    max_path_len = start_r * 2
    
    # 搜索参数
    search_radius_max = 20
    search_radius_min = 3
    max_brightness_increase = 40
    path_brightness_tolerance = 5
    search_sector_angle = 22.0

    # 用于在追踪失败时，返回找到的最长路径
    longest_failed_path = [start_point]

    # 3. BFS主循环
    while q:
        current_point, path = q.popleft()
        px, py = current_point
        
        # 更新最长失败路径
        if len(path) > len(longest_failed_path):
            longest_failed_path = path

        # 检查是否成功到达中心Die区域
        dx, dy, dw, dh = die_rect
        if dx <= px < dx + dw and dy <= py < dy + dh:
            for p in path: visited[p[1], p[0]] = 1
            return path, True, None

        # 检查路径是否过长
        if len(path) >= max_path_len:
            continue

        # --- 动态定义参考向量 ---
        if len(path) == 1:
            vec_ref = (second_anchor[0] - start_point[0], second_anchor[1] - start_point[1])
        else:
            p_hist_far = path[-2]
            vec_ref = (current_point[0] - p_hist_far[0], current_point[1] - p_hist_far[1])

        # 在一个正方形区域内迭代，然后用距离和角度筛选
        for nx in range(px - search_radius_max, px + search_radius_max + 1):
            for ny in range(py - search_radius_max, py + search_radius_max + 1):
                if nx == px and ny == py: continue

                # --- 硬性约束 ---
                # 1. 扇形区域约束
                dist_sq = (nx - px)**2 + (ny - py)**2
                if not (search_radius_min**2 <= dist_sq <= search_radius_max**2):
                    continue
                
                vec_search = (nx - px, ny - py)
                dot_product = vec_ref[0] * vec_search[0] + vec_ref[1] * vec_search[1]
                mag_ref = math.sqrt(vec_ref[0]**2 + vec_ref[1]**2)
                mag_search = math.sqrt(vec_search[0]**2 + vec_search[1]**2)
                if mag_ref > 0 and mag_search > 0:
                    cos_angle = max(-1.0, min(1.0, dot_product / (mag_ref * mag_search)))
                    angle_from_ref = math.degrees(math.acos(cos_angle))
                    if angle_from_ref > search_sector_angle:
                        continue

                # 2. 其他约束
                if not (0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]): continue
                if (nx, ny) in temp_visited: continue
                if image[ny, nx] >= threshold + 10: continue
                if image[ny, nx] > image[py, px] + max_brightness_increase: continue

                # 3. 路径清空检查
                path_is_clear = True
                line_pixels = get_line_pixels(current_point, (nx, ny))
                if len(line_pixels) > 2:
                    bright_points = 0
                    for (lx, ly) in line_pixels[1:-1]:
                        if image[lx, ly] > image[py, px] + path_brightness_tolerance:
                            bright_points += 1
                    if bright_points / (len(line_pixels) - 2) > 0.4:
                        path_is_clear = False
                if not path_is_clear: continue

                # 4. 如果所有约束都通过，则加入队列
                temp_visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                q.append(((nx, ny), new_path))
    
    # 如果队列为空，则追踪失败
    for p in longest_failed_path: visited[p[1], p[0]] = 1
    return longest_failed_path, False, longest_failed_path[-1]


def main(image_path):
    """主函数"""
    start_time = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"错误：无法加载图像 {image_path}")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    height, width = image.shape
    center = (width // 2, height // 2)

    die_size = width / 2.2
    die_rect = (center[0] - die_size // 2, center[1] - die_size // 2, die_size, die_size)

    # 1. 寻找锚点对
    anchor_pairs, threshold, start_r = find_anchors(image, center)

    if not anchor_pairs:
        print("未能找到任何锚点对，程序退出。")
        return

    # 为可视化和追踪准备数据
    anchors = [p[0] for p in anchor_pairs]
    second_anchors = [p[1] for p in anchor_pairs]

    # 2. 从每个锚点开始追踪
    visited = np.zeros_like(image, dtype=np.uint8)
    successful_paths = []
    broken_paths = []
    break_points = []

    for anchor, second_anchor in anchor_pairs:
        if visited[anchor[1], anchor[0]]:
            continue
        
        path, is_successful, break_point = trace_wire_bfs(
            image, anchor, second_anchor, center, die_rect, visited, threshold, start_r
        )
        
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
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    end_time = time.time()
    print (f"耗时: {end_time - start_time:.2f}s")
    visualize_results(original_image, successful_paths, broken_paths, anchors, second_anchors, break_points)

if __name__ == '__main__':
    image_file = '8_mod.png'
    main(image_file)
