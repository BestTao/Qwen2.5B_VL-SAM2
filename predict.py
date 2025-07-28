# 在test_Qwen5的基础上，修改点提示的生成, 比如一个目标可能有多个点，需要使用2，3，4，5，而不全是1
# SAM2接受的labels是[1,1,1,1]的，不能是[2,3]
# 修改了提示词模板，千问结果处理，合并函数。
import json
import torch
from torch_mlu.utils.model_transfer import transfer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import numpy as np
import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
# from sam2.build_sam import build_sam2,build_sam2_video_predictor
import torch_mlu
import torch_mlu.utils.gpu_migration#注意前三行
from sam2.build_sam import build_sam2_video_predictor
# torch.bfloat16=torch.float16
from tqdm import tqdm
import decord
import os
from qwen_vl_utils import process_vision_info  # 官方工具函数
import re
from collections import defaultdict


def generate_points_with_qwen(model, processor, image, user_prompot):
    """
    使用 Qwen2.5-VL 模型生成指定对象的边界框（点提示）。
    通过更明确的提示词，旨在获取规范的 JSON 输出。

    Args:
        model: 已加载的Qwen2.5-VL模型
        processor: 已加载的处理器
        image (PIL.Image): 待检测的RGB图像
        user_prompot (str or list): 需要检测的对象类别列表（如["car", "person"]）或单个字符串。

    Returns:
        list: Qwen-VL 的原始输出列表，通常包含一个字符串（JSON格式）。
    """

    # Ensure user_prompot is a string for the prompt
    if isinstance(user_prompot, list):
        # Join multiple prompts, e.g., ["car", "person"] -> "car或person"
        user_prompot_str = "、".join(user_prompot)
    else:
        user_prompot_str = user_prompot

    # # Create messages
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": image},
    #             {
    #                 "type": "text",
    #                 "text": f"请从图片中检测所有的{user_prompot}。对于每一个检测到的{user_prompot}，根据目标大小，请生成 **1到5个代表其完整轮廓的点**。这些点应该分布在目标的各个部分，例如车头、车身和车尾等。将结果以 JSON 格式输出，其中每个对象的 'points' 键的值必须是一个包含点坐标的**列表的列表**，例如：[{{\"points\": [[x1, y1], [x2, y2]], \"label\": \"label_name\"}}, {{ \"points\": [[x3, y3], [x4, y4], [x5, y5]], \"label\": \"label_name\"}}]"
    #             },
    #         ],
    #     }
    # ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"请从图片中检测所有的{user_prompot_str}。\n"
                              "对于每一个检测到的目标，请生成 **3到5个代表其完整轮廓的关键点**。\n"
                              "这些点应该分布在目标的各个部分，例如汽车的四个角和中心。\n"
                              "请严格按照以下 JSON 数组格式输出结果。**不要有任何额外文字，只返回 JSON。**\n"
                              "**每个 JSON 对象必须精确地包含 'points' 和 'label' 两个键。**\n"
                              "   - 'points' 的值必须是一个包含 `[x, y]` 坐标对的列表，例如 `[[x1, y1], [x2, y2], [x3, y3]]`。\n"
                              "     **每个点必须是一个独立的 `[x, y]` 子列表，即使只有一个点。**\n"
                              "   - 'label' 的值是对象的类别字符串，例如 `\"car\"` 或 `\"road\"`，**不要包含额外的方括号或引号。**\n"
                              "以下是严格的 JSON 示例（请严格遵循此格式，包括所有逗号和括号）：\n"
                              "```json\n"
                              "[\n"
                              "  {\"points\": [[100, 200], [150, 250], [100, 250], [150, 200]], \"label\": \"car\"},\n"
                              "  {\"points\": [[300, 400]], \"label\": \"road\"}  \n" # Example with a single point (still as [[x,y]])
                              "]\n"
                              "```\n"
                              "**再次强调：在 'points' 列表结束的 `]` 之后，必须紧跟一个逗号 `,`，然后才是 'label' 键。**"
                },
            ],
        }
    ]    
    # Execute inference
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Assuming process_vision_info is defined elsewhere and works correctly
    # You'll need to make sure this function is correctly imported or defined.
    # For now, I'll put a placeholder if it's not provided.
    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except NameError:
        print("Warning: 'process_vision_info' not found. Assuming simple image input structure.")
        # Fallback if process_vision_info is not provided
        # This part depends heavily on how your 'messages' are structured for processor
        # For typical single image input, it might look like this:
        image_inputs = [image] # Pass the PIL image directly if processor handles it
        video_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate output
    print("------------------Qwen2.5b_vl正在生成输出------------------")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048
        
        # Potentially add temperature=0.1 or top_p=0.9 to encourage more deterministic output
        # num_beams=1 # Or increase for more diverse, but potentially slower, generation
    )

    # Decode generated IDs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def pro_vl_results(rl_output):
    """
    处理 Qwen-VL 的原始输出，提取 JSON 内容，并将其格式化为 SAM2 所需的点提示。
    增强了对 'points' 嵌套结构、'label' 字符串化列表的鲁棒性，并过滤掉负数坐标点。

    Args:
        rl_output (list): Qwen-VL 模型的原始输出列表，通常包含一个字符串。

    Returns:
        dict: 包含 'points' 和 'label' 键的字典。
              'points' 是一个包含所有对象点的列表，每个子列表是 [[x, y], ...]。
              'label' 是一个包含每个点集对应标签长度的列表，例如 [[N1], [N2], ...]。
    """
    processed_points_for_sam2 = {'points': [], 'label': []}
    
    # 1. 提取 JSON 内容
    json_content_str = None
    if rl_output and isinstance(rl_output[0], str):
        json_pattern = r'```json\n(.*?)\n```'
        json_match = re.search(json_pattern, rl_output[0], re.DOTALL)
        
        if json_match:
            extracted_json_content = json_match.group(1)
            # 尝试修复 ']]"label"' 变成 ']], "label"' 的情况
            repaired_json_content = re.sub(r'\]\]\s*"label"', r']], "label"', extracted_json_content)
            repaired_json_content = re.sub(r'\]\s*"label"', r'], "label"', repaired_json_content)
            
            try:
                raw_point_results = json.loads(repaired_json_content)
            except json.JSONDecodeError as e:
                print(f"警告: 提取的 JSON 内容解析失败: {e}. 内容: {repaired_json_content[:200]}...")
                raw_point_results = []
        else:
            # 如果没有 ```json``` 标记，尝试直接解析，尽管Qwen-VL通常会带
            try:
                raw_point_results = json.loads(rl_output[0])
            except json.JSONDecodeError:
                print("未在 ```json 标记之间或作为纯 JSON 字符串找到 JSON 内容。")
                raw_point_results = []
    else:
        print(f"警告: Qwen-VL 输出格式异常: {rl_output}. 期望一个包含字符串的列表。")
        raw_point_results = []

    # 确保 raw_point_results 是一个列表，以便统一处理
    if not isinstance(raw_point_results, list):
        raw_point_results = [raw_point_results]

    # 2. 遍历解析后的结果，提取并格式化点信息
    for obj_data in raw_point_results:
        if 'points' in obj_data and isinstance(obj_data['points'], list):
            current_obj_raw_points = obj_data['points']
            
            # --- 扁平化 points 列表 START ---
            flattened_points = []
            def flatten(items):
                """递归扁平化列表，直到遇到非列表元素或[x, y]对"""
                for item in items:
                    if isinstance(item, list) and len(item) == 2 and all(isinstance(coord_val, (int, float)) for coord_val in item):
                        # --- 新增的负数坐标过滤逻辑 START ---
                        if item[0] >= 0 and item[1] >= 0: # 确保 x 和 y 坐标都非负
                            # 找到了一个 [x, y] 对，且坐标有效，直接添加
                            flattened_points.append(item)
                        else:
                            print(f"警告: 发现负数或无效坐标，已过滤: {item}")
                        # --- 新增的负数坐标过滤逻辑 END ---
                    elif isinstance(item, list):
                        # 如果是列表，继续递归扁平化
                        flatten(item)
                    # 否则，跳过非列表或非 [x, y] 的项，或可以添加警告
            
            flatten(current_obj_raw_points)
            # --- 扁平化 points 列表 END ---

            # --- 解析 label 字段 START ---
            extracted_label = "unknown" # 默认值
            raw_label = obj_data.get('label', '')
            if isinstance(raw_label, str):
                # 尝试解析 "['road']" 或 "road"
                try:
                    # 将单引号替换为双引号以便JSON解析
                    parsed_label_list = json.loads(raw_label.replace("'", '"')) 
                    if isinstance(parsed_label_list, list) and len(parsed_label_list) > 0:
                        extracted_label = str(parsed_label_list[0]) # 取列表的第一个元素
                    else:
                        extracted_label = raw_label # 如果解析失败或不是列表，就用原始字符串
                except json.JSONDecodeError:
                    extracted_label = raw_label # 如果不是有效的JSON字符串，就用原始字符串
            elif raw_label is not None:
                # 如果 label 直接是字符串（如 "road"），或其他非字符串类型
                extracted_label = str(raw_label)
            # --- 解析 label 字段 END ---

            if flattened_points: # 如果当前对象有有效的点
                processed_points_for_sam2['points'].append(flattened_points)
                # SAM2通常期望每个点一个标签，且都是前景（1）
                # 我们这里存储的是该点集包含的点数量，在merge_annotations中会转化为1
                processed_points_for_sam2['label'].append([len(flattened_points)])
            else:
                print(f"警告: 对象数据中 'points' 字段为空或无效，或者未能成功扁平化: {obj_data}")
        else:
            print(f"警告: 对象数据中缺少 'points' 键或其格式不正确: {obj_data}")

    return processed_points_for_sam2

# def pro_vl_results_7B(rl_output):
#     """
#     处理 Qwen-VL 的原始输出，提取 JSON 内容，并将其格式化为 SAM2 所需的点提示。

#     Args:
#         rl_output (list): Qwen-VL 模型的原始输出列表，通常包含一个字符串。

#     Returns:
#         dict: 包含 'points' 和 'label' 键的字典，格式为
#               {'points': [[[x1, y1], [x2, y2], ...], [[x3, y3], ...]], 'label': [[N1], [N2], ...]}
#               其中 N1, N2 是对应对象的点数量。
#     """
#     processed_points_for_sam2 = {'points': [], 'label': []}
#     json_content_str = None

#     # 1. 尝试直接解析 JSON，如果 Qwen-VL 输出了一个合法的 JSON 字符串
#     if rl_output and isinstance(rl_output[0], str):
#         try:
#             # 尝试从字符串中直接加载 JSON。
#             raw_point_results = json.loads(rl_output[0])
#             json_content_str = rl_output[0] # 标记为已成功解析
#         except json.JSONDecodeError:
#             # 如果直接加载失败，尝试用正则表达式提取 ```json``` 块
#             json_pattern = r'```json\n(.*?)\n```'
#             json_match = re.search(json_pattern, rl_output[0], re.DOTALL)
#             if json_match:
#                 json_content_str = json_match.group(1)
#                 try:
#                     raw_point_results = json.loads(json_content_str)
#                 except json.JSONDecodeError as e:
#                     print(f"警告: 提取的 JSON 内容解析失败: {e}. 内容: {json_content_str[:200]}...")
#                     raw_point_results = []
#             else:
#                 print("未在 ```json 标记之间或作为纯 JSON 字符串找到 JSON 内容。")
#                 raw_point_results = []
#     else:
#         print(f"警告: Qwen-VL 输出格式异常: {rl_output}. 期望一个包含字符串的列表。")
#         raw_point_results = []

#     # 确保 raw_point_results 是一个列表，以便统一处理
#     if not isinstance(raw_point_results, list):
#         raw_point_results = [raw_point_results]

#     # 2. 遍历解析后的结果，提取点信息
#     for obj_data in raw_point_results:
#         if 'points' in obj_data and isinstance(obj_data['points'], list):
#             # Qwen-VL 的输出中 'points' 字段是一个包含多个点列表的列表
#             # 例如: [[[x1, y1], [x2, y2]], [[x3, y3]]]
#             for point_list_for_an_object in obj_data['points']:
#                 # 确保 point_list_for_an_object 是一个包含 [x, y] 对的列表
#                 # 我们不再需要 flatten_points 函数，因为 SAM-2 期望的格式就是嵌套的
#                 # 直接使用这个内部列表作为 SAM-2 的一个点集
                
#                 # 简单的校验，确保列表中的元素是 [x, y] 形式
#                 valid_points = []
#                 for p in point_list_for_an_object:
#                     if isinstance(p, list) and len(p) == 2 and all(isinstance(coord, (int, float)) for coord in p):
#                         valid_points.append(p)
                
#                 if valid_points: # 如果当前对象有有效的点
#                     # 将当前对象的点列表作为 SAM-2 的一个独立点集添加
#                     processed_points_for_sam2['points'].append(valid_points)
#                     # 为该点集添加对应的标签，即该点集的点数量
#                     processed_points_for_sam2['label'].append([len(valid_points)])
#         else:
#             print(f"警告: 对象数据中缺少 'points' 键或其格式不正确: {obj_data}")

#     # print(f"为 SAM-2 格式化后的点和标签: {processed_points_for_sam2}")
#     return processed_points_for_sam2


import json
import re

def pro_vl_results_32B(rl_output):
    """
    处理 Qwen-VL 的原始输出，提取 JSON 内容，并将其格式化为 SAM2 所需的点提示。
    增强了对 'points' 嵌套结构、'label' 字符串化列表的鲁棒性，过滤负数坐标，
    并更灵活地修复 JSON 格式中缺失的逗号。

    Args:
        rl_output (list): Qwen-VL 模型的原始输出列表，通常包含一个字符串。

    Returns:
        dict: 包含 'points' 和 'label' 键的字典。
              'points' 是一个包含所有对象点的列表，每个子列表是 [[x, y], ...]。
              'label' 是一个包含每个点集对应标签长度的列表，例如 [[N1], [N2], ...]。
    """
    processed_points_for_sam2 = {'points': [], 'label': []}
    # print(f"😊😊😊😊{rl_output[0]}")
    # 1. 提取 JSON 内容
    json_content_str = None
    if rl_output and isinstance(rl_output[0], str):
        json_pattern = r'```json\n(.*?)\n```'
        json_match = re.search(json_pattern, rl_output[0], re.DOTALL)
        
        if json_match:
            extracted_json_content = json_match.group(1)
            
            # --- 更通用的 JSON 修复逻辑 START ---
            # 修复 '[...]"label"' 变成 '[...], "label"' 的情况
            # 匹配任何一个或多个 ']' 后面直接跟着一个 '"' (如果前面不是 ',')
            # 这是一个更通用的模式，以捕捉 ']]"label"' 和 ']"label"' 甚至可能是 ']]]"label"'
            repaired_json_content = re.sub(r'\]\s*(?<!,)"', r'], "', extracted_json_content)
            # 再次修复，以防有嵌套的 ]] 且之前没有逗号，例如 [[x,y]],"label" 变成 [[x,y]],,"label"
            # 确保只在必要时添加逗号，避免重复
            repaired_json_content = re.sub(r'\]\s*,\s*\]\s*,\s*"', r']], "', repaired_json_content) # For cases like ']],,"label"' from previous repair
            
            repaired_json_content = re.sub(r'"label":\s*"\[\'(.*?)\'\]"', r'"label": "\1"', repaired_json_content)

            # --- 更通用的 JSON 修复逻辑 END ---
            
            try:
                raw_point_results = json.loads(repaired_json_content)
            except json.JSONDecodeError as e:
                print(f"警告: 提取的 JSON 内容解析失败: {e}. 内容: {repaired_json_content[:200]}...")
                raw_point_results = []
        else:
            # If no ```json``` marker, try direct parsing, although Qwen-VL usually provides one
            try:
                raw_point_results = json.loads(rl_output[0])
            except json.JSONDecodeError:
                print("未在 ```json 标记之间或作为纯 JSON 字符串找到 JSON 内容。")
                raw_point_results = []
    else:
        print(f"警告: Qwen-VL 输出格式异常: {rl_output}. Expected a list containing a string.")
        raw_point_results = []

    # Ensure raw_point_results is a list for uniform processing
    if not isinstance(raw_point_results, list):
        raw_point_results = [raw_point_results]

    # 2. Iterate through parsed results, extract and format point information
    for obj_data in raw_point_results:
        if 'points' in obj_data and isinstance(obj_data['points'], list):
            current_obj_raw_points = obj_data['points']
            
            # --- Flatten points list START ---
            flattened_points = []
            def flatten(items):
                """Recursively flattens lists until non-list elements or [x, y] pairs are found."""
                for item in items:
                    # Check for [x, y] pair and non-negative coordinates
                    if isinstance(item, list) and len(item) == 2 and all(isinstance(coord_val, (int, float)) for coord_val in item):
                        if item[0] >= 0 and item[1] >= 0:
                            flattened_points.append(item)
                        else:
                            print(f"警告: 发现负数或无效坐标，已过滤: {item}")
                    elif isinstance(item, list):
                        flatten(item)
                    # Else, skip non-list or non-[x, y] items
            
            flatten(current_obj_raw_points)
            # --- Flatten points list END ---

            # --- Parse label field START ---
            extracted_label = "unknown" # Default value
            raw_label = obj_data.get('label', '')
            # If label is a string, try to clean it
            if isinstance(raw_label, str):
                # We tried to repair `["'road'"]` to `"road"` earlier with regex.
                # If that failed, this attempts to parse it as JSON
                try:
                    parsed_label_list = json.loads(raw_label.replace("'", '"')) 
                    if isinstance(parsed_label_list, list) and len(parsed_label_list) > 0:
                        extracted_label = str(parsed_label_list[0])
                    else:
                        extracted_label = raw_label 
                except json.JSONDecodeError:
                    extracted_label = raw_label # If not a valid JSON string, use original
            elif raw_label is not None:
                extracted_label = str(raw_label)
            # --- Parse label field END ---

            if flattened_points: # If current object has valid points
                processed_points_for_sam2['points'].append(flattened_points)
                # SAM2 usually expects one label per point, all foreground (1)
                # We store the number of points in this set; it will be converted to 1s in merge_annotations
                processed_points_for_sam2['label'].append([len(flattened_points)]) # Keep the length for now
            else:
                print(f"警告: 对象数据中 'points' 字段为空或无效，或者未能成功扁平化: {obj_data}")
        else:
            print(f"警告: 对象数据中缺少 'points' 键或其格式不正确: {obj_data}")

    return processed_points_for_sam2
def timeit(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result
    return wrapper

# def merge_annotations(annotations):
#     """
#     合并具有相同 frame_idx 的注解。
#     将每个 frame_idx 下的所有点的 'points' 和 'labels' 聚合起来。

#     Args:
#         annotations (list): 原始的注解列表，每个元素包含 'obj_id', 'frame_idx', 'points', 'labels', 'box'。

#     Returns:
#         list: 合并后的注解列表。每个元素代表一个 frame_idx 下的所有对象。
#               其 'points' 包含所有对象的扁平化点列表，'labels' 包含所有原始对象的点数量列表。
#               'obj_id' 表示该帧中原始对象的总数。
#     """
#     merged_by_frame = defaultdict(
#         lambda: {'points': [], 'labels': [], 'box': [], 'original_obj_count': 0}
#     )

#     for annot in annotations:
#         frame_idx = annot['frame_idx']
        
#         # 将当前注解的点添加到该帧的总点列表中 (扁平化)
#         # 确保 annot['points'] 是一个列表的列表，每个子列表是 [x, y]
#         # 您的输入示例是：'points': [[1600, 1080], [1790, 1080], [1850, 1080]]
#         # 这里直接 extend 即可
#         merged_by_frame[frame_idx]['points'].extend(annot['points'])
        
#         # 将原始对象的标签（点数量）添加到该帧的总标签列表中
#         # 确保 annot['labels'] 是一个像 [3] 这样的列表
#         merged_by_frame[frame_idx]['labels'].extend(annot['labels'])
        
#         # 增加原始对象计数
#         merged_by_frame[frame_idx]['original_obj_count'] += 1
        
#         # 框信息（如果存在，这里简单地取第一个，或者根据需要进行合并）
#         # 考虑到你的示例中 box 都是空列表，这里不做复杂合并
#         if annot['box']:
#             # 这里的逻辑需要根据你实际的 box 处理需求来定
#             # 如果是多个 box 需要合并成一个大 box，需要更复杂的计算
#             # 如果只是为了占位，可以保持空
#             pass # 或者 merged_by_frame[frame_idx]['box'].extend(annot['box'])

#     final_merged_annotations = []
#     for frame_idx in sorted(merged_by_frame.keys()):
#         merged_data = merged_by_frame[frame_idx]
#         final_merged_annotations.append({
#             'obj_id': merged_data['original_obj_count'], # 'obj_id'表示该帧中原始对象的总数
#             'frame_idx': frame_idx,
#             'points': merged_data['points'], # 包含所有原始对象的扁平化点列表
#             'labels': merged_data['labels'], # 包含所有原始对象的点数量列表，如 [3, 3, 3]
#             'box': merged_data['box'] # 保持为原始收集的box（如果存在）
#         })
#     return final_merged_annotations
def merge_annotations(annotations):
    """
    合并具有相同 frame_idx 的注解。
    将每个 frame_idx 下的所有点的 'points' 和 'labels' 聚合起来。
    生成的 'labels' 格式为 [1, 1, ..., 1]，数量与合并后的点数匹配。

    Args:
        annotations (list): 原始的注解列表，每个元素包含 'obj_id', 'frame_idx', 'points', 'labels', 'box'。

    Returns:
        list: 合并后的注解列表。每个元素代表一个 frame_idx 下的所有对象。
              其 'points' 包含所有对象的扁平化点列表，'labels' 包含与这些点一一对应的 '1' 标签。
              'obj_id' 表示该帧中原始对象的总数（用于元数据，SAM2可能不直接使用此ID）。
    """
    merged_by_frame = defaultdict(
        lambda: {'points': [], 'labels': [], 'box': [], 'original_obj_count': 0}
    )

    for annot in annotations:
        frame_idx = annot['frame_idx']
        
        # 将当前注解的点添加到该帧的总点列表中 (扁平化)
        # annot['points'] 已经是像 [[x1, y1], [x2, y2]] 这样的列表
        merged_by_frame[frame_idx]['points'].extend(annot['points'])
        
        # 根据当前注解的点数量，添加相应数量的 '1' 到该帧的总标签列表中
        # annot['points'] 的长度就是该原始对象所包含的点数
        num_points_in_current_object = len(annot['points'])
        merged_by_frame[frame_idx]['labels'].extend([1] * num_points_in_current_object)
        
        # 增加原始对象计数
        merged_by_frame[frame_idx]['original_obj_count'] += 1
        
        # 框信息（如果存在，这里简单地取第一个，或者根据需要进行合并）
        if annot['box']:
            # 这里的逻辑需要根据你实际的 box 处理需求来定
            pass # 保持不动，或根据你的合并策略处理

    final_merged_annotations = []
    for frame_idx in sorted(merged_by_frame.keys()):
        merged_data = merged_by_frame[frame_idx]
        final_merged_annotations.append({
            'obj_id': merged_data['original_obj_count'], # 此处的 obj_id 作为元数据，表示原始对象的总数
            'frame_idx': frame_idx,
            'points': merged_data['points'], # 包含所有原始对象的扁平化点列表
            'labels': merged_data['labels'], # 包含与所有点一一对应的 '1' 标签，如 [1, 1, 1, 1, 1, 1, 1, 1, 1]
            'box': merged_data['box'] # 保持为原始收集的box（如果存在）
        })
    return final_merged_annotations

def init_models():
    """初始化所有模型"""
    # SAM2
    SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    
    return sam_predictor

class ObjectTracker:
    def __init__(self, iou_threshold=0.3, max_missing_frames=30):
        self.objects = {}  # obj_id: {"last_box", "last_frame", "prompt"}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing_frames
    
    def update(self, detections, frame_idx):
        """更新追踪器，返回每个检测对应的对象ID"""
        self._cleanup(frame_idx)
        
        if not detections:
            return []
        
        # New: Initialize the list to store obj_id for each detection
        # This list will have the same length and order as 'detections'
        assigned_obj_ids = [None] * len(detections)
        
        # Keep track of which existing objects have been matched in this frame
        matched_existing_obj_ids = set()
        
        # 1. 尝试匹配已有对象
        for obj_id, obj_data in list(self.objects.items()): # Use list() to iterate over a copy if modifying self.objects
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                # Skip if this detection has already been assigned an ID
                if assigned_obj_ids[det_idx] is not None:
                    continue
                
                # Only match objects with the same prompt (important for multi-object tracking)
                if det["prompt"] != obj_data["prompt"]:
                    continue
                
                iou = self._calculate_iou(obj_data["last_box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_iou > self.iou_threshold:
                # Match successful: Assign the existing obj_id to this detection
                assigned_obj_ids[best_det_idx] = obj_id
                matched_existing_obj_ids.add(obj_id) # Mark existing obj as matched
                
                # Update object state with the new box and current frame
                self.objects[obj_id]["last_box"] = detections[best_det_idx]["box"]
                self.objects[obj_id]["last_frame"] = frame_idx
        
        # 2. 处理未匹配的检测（新对象）
        for det_idx, det in enumerate(detections):
            if assigned_obj_ids[det_idx] is None: # If this detection has not been assigned an ID yet
                new_id = self.next_id
                self.next_id += 1
                self.objects[new_id] = {
                    "prompt": det["prompt"],
                    "last_box": det["box"],
                    "last_frame": frame_idx
                }
                assigned_obj_ids[det_idx] = new_id
        
        # 3. Update 'last_frame' for existing objects that were not matched in this frame
        # (This is implicitly handled by _cleanup, but explicitly setting for robustness if _cleanup logic changes)
        # However, the current cleanup handles it fine. Just ensuring all objects in self.objects have updated 'last_frame' or are marked for cleanup.
        # This loop is technically not needed if _cleanup handles it and assigned_obj_ids fully covers all 'detections'

        return assigned_obj_ids # Return the list with obj_ids for each detection

    def _cleanup(self, current_frame):
        """清理长时间未出现的对象"""
        to_remove = []
        for obj_id, obj_data in list(self.objects.items()): # Iterate over a copy
            if current_frame - obj_data["last_frame"] > self.max_missing:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.objects[obj_id]
    
    def _calculate_iou(self, box1, box2):
        """计算两个框的IoU"""
        # ... (IoU calculation remains the same)
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

CONFIG = {
    "prompts": ["house"], # 提示词
    "keyframe_interval": 20,   # 提示帧的频率
    "sam_config": "checkpoints/sam2.1_hiera_large.pt",
    "sam_checkpoint": "configs/sam2.1/sam2.1_hiera_l.yaml"
}

def main():
    # 0. 配置
    video_file = "/workspace/volume/gxs2/zht/project/sam2-sam2.1_mlu/notebooks/videos/video2-12s.mp4"
    output_dir = "/workspace/volume/gxs2/zht/project/sam2-sam2.1_mlu/notebooks/videos/house"
    # 1. 初始化SAM2模型
    sam_predictor= init_models()
    
    # 2. 准备视频
    vr = decord.VideoReader(video_file)
    total_frames = len(vr)
    # keyframe_indices = set()
    # keyframe_indices.add(0)
    # 3. 加载Qwen模型（仅需加载一次）
    # model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/workspace/volume/gxs2/zht/Models/models/Qwen/Qwen2.5-VL-32B-Instruct",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    qwen_processor = AutoProcessor.from_pretrained(
        "/workspace/volume/gxs2/zht/Models/models/Qwen/Qwen2.5-VL-32B-Instruct"
        )

    # 4. 确定关键帧
    keyframe_indices = set([0]) # 保持为set类型
    
    # 第一次添加关键帧
    for frame_idx in range(0, total_frames, CONFIG["keyframe_interval"]):
        keyframe_indices.add(frame_idx)
    # 在所有添加操作完成后，再将 set 转换为排序后的 list
    keyframe_indices = sorted(list(keyframe_indices)) # 先转换为list再排序
    print(f"关键帧列表: {keyframe_indices}")

    # 5. 收集提示点
    annotations = []
    tracked_objects = {}
    next_obj_id = 1
    object_tracker = ObjectTracker(
        iou_threshold=0.3,   # IoU匹配阈值
        max_missing_frames=30 # 最大消失帧数
    )    
    
    for frame_idx in tqdm(keyframe_indices, desc="处理关键帧"):
        # 获取并处理帧
        frame = vr[frame_idx].asnumpy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil_from_array = Image.fromarray(frame_rgb)
        objects_to_detect = CONFIG["prompts"]
        # if frame_idx == 0:
        #     objects_to_detect = CONFIG["prompts"]
        # else:
        #     objects_to_detect = [obj["prompt"] for obj in tracked_objects.values()]

        
        # # image_path = "demo.jpeg"
        # # frame_pil = Image.open(image_path)
        print(f"objects_to_detect:{objects_to_detect}")
        # 使用Qwen模型生成点提示
        vl_results = generate_points_with_qwen(
            model=qwen_model,
            processor=qwen_processor,
            image=frame_pil_from_array,
            user_prompot=objects_to_detect
        )
        print(f"vl_results:{vl_results}")
        # processed_points = pro_vl_results(vl_results)
        processed_points = pro_vl_results_32B(vl_results)
        print(f"处理后的LLM结果:{processed_points}")

        # 直接遍历 processed_points['points'] 中每个独立的点集
        next_obj_id = 1 # 确保每次处理新帧时，obj_id从1开始递增，或者有更完善的全局ID管理
        for i, point_set in enumerate(processed_points['points']):
            # point_set 是一个列表，例如 [[1600, 1080], [1790, 1080], [1850, 1080]]
            # 这里的 obj_id 需要考虑是针对当前帧的新对象，还是跨帧追踪的旧对象
            # 如果不进行跨帧跟踪（如您所说，Qwen-VL避免同一对象），则每检测到一个新点集，就给一个新ID
            obj_id = next_obj_id
            next_obj_id += 1 # 每次检测到一个新的对象（点集）就递增ID
            
            # 假设 Qwen-VL 返回的 processed_points['label'] 与 'points' 的子列表一一对应
            # 例如 processed_points['label'][i] 对应 processed_points['points'][i]
            label_for_this_set = processed_points['label'][i] # 应该是一个像 [[3]] 这样的列表

            annotations.append({
                "obj_id": obj_id, # 为每个Qwen-VL检测到的对象（点集）分配一个独立的ID
                "frame_idx": frame_idx,
                "points": point_set, # 这是一个包含多个 [x, y] 的列表
                "labels": label_for_this_set, # 这是该点集对应的标签，例如 [[3]]
                "box": []  # 没有边界框信息，保持为空列表
            })

    # annotations = [{'obj_id': 1, 'frame_idx': 0, 'points': [[1568, 267]], 'labels': [[1]], 'box': [1540, 223, 1596, 311]}, {'obj_id': 1, 'frame_idx': 0, 'points': [[1751, 85]], 'labels': [[1]], 'box': [1728, 46, 1775, 125]}, {'obj_id': 1, 'frame_idx': 0, 'points': [[2256, 40]], 'labels': [[1]], 'box': [2240, 29, 2272, 52]}, {'obj_id': 1, 'frame_idx': 50, 'points': [[1673, 273]], 'labels': [[1]], 'box': [1645, 239, 1702, 308]}, {'obj_id': 1, 'frame_idx': 50, 'points': [[1830, 518]], 'labels': [[1]], 'box': [1808, 499, 1852, 538]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[138, 175]], 'labels': [[1]], 'box': [109, 143, 167, 208]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[142, 362]], 'labels': [[1]], 'box': [115, 328, 170, 397]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[141, 447]], 'labels': [[1]], 'box': [115, 400, 167, 494]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[449, 751]], 'labels': [[1]], 'box': [400, 728, 499, 775]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[462, 879]], 'labels': [[1]], 'box': [409, 858, 516, 900]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[465, 1018]], 'labels': [[1]], 'box': [415, 996, 516, 1040]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[489, 1305]], 'labels': [[1]], 'box': [428, 1258, 550, 1352]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[168, 1215]], 'labels': [[1]], 'box': [142, 1158, 195, 1272]}, {'obj_id': 1, 'frame_idx': 150, 'points': [[294, 1357]], 'labels': [[1]], 'box': [232, 1314, 357, 1400]}, {'obj_id': 1, 'frame_idx': 200, 'points': [[110, 417]], 'labels': [[1]], 'box': [81, 329, 140, 506]}, {'obj_id': 1, 'frame_idx': 200, 'points': [[111, 571]], 'labels': [[1]], 'box': [87, 532, 136, 610]}, {'obj_id': 1, 'frame_idx': 200, 'points': [[112, 649]], 'labels': [[1]], 'box': [89, 588, 136, 710]}, {'obj_id': 1, 'frame_idx': 200, 'points': [[425, 1160]], 'labels': [[1]], 'box': [352, 968, 498, 1352]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[405, 1301]], 'labels': [[1]], 'box': [341, 1265, 470, 1338]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 1238]], 'labels': [[1]], 'box': [339, 1199, 470, 1278]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 1164]], 'labels': [[1]], 'box': [339, 1124, 470, 1204]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 1096]], 'labels': [[1]], 'box': [339, 1058, 470, 1134]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 1026]], 'labels': [[1]], 'box': [339, 988, 470, 1064]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 958]], 'labels': [[1]], 'box': [339, 920, 470, 996]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 892]], 'labels': [[1]], 'box': [339, 854, 470, 930]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 826]], 'labels': [[1]], 'box': [339, 788, 470, 864]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 760]], 'labels': [[1]], 'box': [339, 722, 470, 798]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 694]], 'labels': [[1]], 'box': [339, 656, 470, 732]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 626]], 'labels': [[1]], 'box': [339, 589, 470, 664]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 561]], 'labels': [[1]], 'box': [339, 523, 470, 600]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 495]], 'labels': [[1]], 'box': [339, 457, 470, 533]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 429]], 'labels': [[1]], 'box': [339, 391, 470, 467]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 363]], 'labels': [[1]], 'box': [339, 325, 470, 401]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 297]], 'labels': [[1]], 'box': [339, 259, 470, 335]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 230]], 'labels': [[1]], 'box': [339, 193, 470, 267]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 163]], 'labels': [[1]], 'box': [339, 127, 470, 199]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 93]], 'labels': [[1]], 'box': [339, 60, 470, 127]}, {'obj_id': 1, 'frame_idx': 250, 'points': [[404, 30]], 'labels': [[1]], 'box': [339, 0, 470, 60]}, {'obj_id': 1, 'frame_idx': 300, 'points': [[393, 1192]], 'labels': [[1]], 'box': [340, 1156, 447, 1228]}, {'obj_id': 1, 'frame_idx': 300, 'points': [[403, 1277]], 'labels': [[1]], 'box': [340, 1239, 467, 1316]}, {'obj_id': 1, 'frame_idx': 300, 'points': [[403, 1366]], 'labels': [[1]], 'box': [340, 1328, 467, 1404]}, {'obj_id': 1, 'frame_idx': 300, 'points': [[70, 792]], 'labels': [[1]], 'box': [44, 711, 97, 874]}, {'obj_id': 1, 'frame_idx': 300, 'points': [[70, 643]], 'labels': [[1]], 'box': [44, 599, 97, 688]}, {'obj_id': 1, 'frame_idx': 300, 'points': [[70, 515]], 'labels': [[1]], 'box': [44, 471, 97, 560]}]

    # print(f"得到的annotations：{annotations}")
    annotations = merge_annotations(annotations)
    print(f"合并后的annotations：{annotations}")
    # annotations = [{'obj_id': 7, 'frame_idx': 0, 'points': [[168, 479], [358, 479], [358, 748], [168, 748], [168, 479], [1348, 10], [1680, 10], [1680, 100], [1348, 100], [1348, 10], [1348, 130], [1680, 130], [1680, 260], [1348, 260], [1348, 130], [1348, 290], [1680, 290], [1680, 420], [1348, 420], [1348, 290], [1348, 450], [1680, 450], [1680, 580], [1348, 580], [1348, 450], [1348, 610], [1680, 610], [1680, 740], [1348, 740], [1348, 610], [1348, 770], [1680, 770], [1680, 900], [1348, 900], [1348, 770]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 3, 'frame_idx': 20, 'points': [[167, 489], [388, 489], [388, 740], [167, 740], [167, 489], [1310, 168], [1680, 168], [1680, 388], [1310, 388], [1310, 168], [1310, 500], [1680, 500], [1680, 719], [1310, 719], [1310, 500]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 1, 'frame_idx': 40, 'points': [[168, 479], [388, 686], [388, 1078], [168, 1078]], 'labels': [1, 1, 1, 1], 'box': []}, {'obj_id': 2, 'frame_idx': 60, 'points': [[16, 178], [498, 198], [498, 378], [16, 378], [16, 400], [418, 1088], [418, 400]], 'labels': [1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 4, 'frame_idx': 80, 'points': [[136, 187], [549, 260], [549, 378], [136, 378], [136, 580], [549, 660], [549, 1080], [136, 1080], [1308, 18], [1570, 120], [1570, 170], [1308, 170], [1200, 170], [1570, 300], [1570, 500], [1200, 500]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 2, 'frame_idx': 100, 'points': [[16, 187], [598, 610], [388, 1068], [16, 187], [1200, 10], [1578, 386], [1578, 750], [1200, 10]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 2, 'frame_idx': 120, 'points': [[136, 378], [590, 378], [590, 826], [136, 826], [1298, 200], [1536, 200], [1536, 576], [1298, 576]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 2, 'frame_idx': 140, 'points': [[136, 478], [598, 478], [598, 886], [136, 886], [1298, 260], [1560, 260], [1560, 606], [1298, 606]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 2, 'frame_idx': 160, 'points': [[136, 789], [588, 368], [588, 789], [136, 789], [1300, 340], [1596, 340], [1596, 640], [1300, 640]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 3, 'frame_idx': 180, 'points': [[1368, 79], [1458, 116], [1368, 116], [1458, 79], [1340, 278], [1570, 694], [1340, 694], [1570, 278], [1, 358], [608, 996], [1, 996], [608, 358]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 3, 'frame_idx': 200, 'points': [[1368, 379], [1578, 510], [1578, 688], [1368, 688], [1248, 100], [1498, 160], [1498, 240], [1248, 240], [1, 460], [598, 650], [598, 1070], [1, 1070]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 3, 'frame_idx': 220, 'points': [[1368, 79], [1488, 188], [1368, 188], [1488, 79], [1308, 350], [1665, 808], [1308, 808], [1665, 350], [1, 450], [588, 1086], [1, 1086], [588, 450]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 3, 'frame_idx': 240, 'points': [[1378, 69], [1498, 180], [1498, 260], [1378, 260], [1340, 390], [1686, 880], [1686, 620], [1340, 620], [0, 520], [570, 1080], [570, 520], [0, 1080]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}, {'obj_id': 4, 'frame_idx': 260, 'points': [[168, 79], [800, 100], [800, 200], [168, 200], [1318, 100], [1500, 100], [1500, 200], [1318, 200], [1340, 450], [1700, 450], [1700, 950], [1340, 950], [1, 560], [580, 560], [580, 1080], [1, 1080]], 'labels': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'box': []}]
    
    # --- Qwen-VL 模型推理完成，现在释放显存 ---
    # 1. 将模型移到CPU（或任何其他不会占用的设备）
    qwen_model.cpu() 
    
    # 2. 删除模型对象及其引用
    del qwen_model 
    del qwen_processor # 处理器通常占用较少，但最好也清理
    
    # 3. 清理PyTorch的显存缓存
    # 确保所有未被引用的张量占用的显存被释放回系统
    if torch.mlu.is_available(): # 或者 torch.cuda.is_available() for GPU
        torch.mlu.empty_cache() # 或者 torch.cuda.empty_cache()
    print("--------------------- Qwen-VL 模型显存已释放---------------------")
    # --- 显存释放结束 ---

    # 6. 使用SAM2进行视频分割（与之前相同）
    inference_state = sam_predictor.init_state(video_path=video_file)
    sam_predictor.reset_state(inference_state)
    video_segments = {}
    with torch.inference_mode(), torch.autocast("mlu", dtype=torch.float16):
        # 添加所有提示
        for annot in tqdm(annotations, desc="添加提示"):
            # 添加点提示（仅主实例）
            points_tensor = torch.tensor(annot["points"])
            # print(f"points_tensor:{points_tensor}")
            labels_tensor = torch.tensor(annot["labels"])
            # print(f"labels_tensor:{labels_tensor}")

            # 添加框提示,后续需要修改
            box_tensor = torch.tensor(annot["box"][0]) if annot["box"] else None
            
            sam_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=annot["frame_idx"],
                obj_id=annot["obj_id"],
                points=points_tensor,
                labels=labels_tensor,
                box=None
            )
        
        # 传播并获取分割结果
        for out_frame_idx, out_obj_ids, out_mask_logits in sam_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                # 获取掩码并确保是二维的
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                if mask.ndim == 3:  # 如果是(1, H, W)形状
                    mask = mask[0]  # 取出第一维
                elif mask.ndim > 3:  # 如果是更高维度
                    mask = mask.squeeze()  # 压缩所有单维度
                
                # 确保最终是二维的
                if mask.ndim != 2:
                    print(f"警告: 掩码维度异常 {mask.shape}，尝试自动修正")
                    mask = mask.squeeze()  # 再次尝试压缩
                    if mask.ndim != 2:
                        raise ValueError(f"无法修正掩码维度: {mask.shape}")
                
                video_segments[out_frame_idx][out_obj_id] = mask

    # --- （使用指定颜色） ---
    obj_colors = {}
    # 假设我们选择一个明亮的蓝色
    fixed_color_bgr = (255, 0, 0) # B, G, R (蓝色分量最高)
    # 遍历所有对象ID，并为它们分配相同的固定颜色
    # 你仍然需要收集 all_obj_ids，因为需要知道有哪些对象ID会出现在 video_segments 中
    all_obj_ids = set()
    for seg in video_segments.values():
        for obj_id in seg.keys():
            all_obj_ids.add(obj_id)
            
    for obj_id in all_obj_ids:
        obj_colors[obj_id] = fixed_color_bgr

    vis_frame_stride = 1
    # 7保存结果
    frame_indices = range(0, total_frames, vis_frame_stride)
    image_paths = []  # 存储所有图像路径
    log_interval = max(1, len(frame_indices) // 20)  # 总共约20条日志

    for i, out_frame_idx in enumerate(tqdm(frame_indices, desc="Saving results")):
        frame = vr[out_frame_idx].numpy()
        output_path = os.path.join(output_dir, f"seg_{out_frame_idx:04d}.jpg")
        
        # 如果有分割结果
        if out_frame_idx in video_segments:
            masks = video_segments[out_frame_idx]
            
            # 创建结果帧
            result_frame = frame.copy()
            
            # 应用所有掩码
            for obj_id, mask in masks.items():
                # 检查掩码形状是否匹配
                if mask.shape != frame.shape[:2]:
                    print(f"警告: 帧 {out_frame_idx} 的掩码形状 {mask.shape} 与图像形状 {frame.shape[:2]} 不匹配，正在调整")
                    mask = cv2.resize(mask.astype(np.float32), 
                                    (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                    mask = mask > 0.5  # 重新二值化
                
                color = obj_colors[obj_id]
                
                # 高效应用掩码
                masked_area = result_frame[mask]

                if masked_area.size > 0:  # 确保有区域需要处理
                    color_np = np.array(color, dtype=result_frame.dtype).reshape(1, 3)
                    blended = (masked_area * 0.7 + color_np * 0.3).astype(result_frame.dtype)
                    result_frame[mask] = blended
        
        else:
            result_frame = frame
        
        # 保存为JPEG
        cv2.imwrite(output_path, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
        image_paths.append(output_path)
        
        # 减少日志频率
        # if i % log_interval == 0:
        #     print(f"Saved frame {out_frame_idx} to {output_path}")

    # 创建视频
    if image_paths:
        # 创建视频输出路径
        video_output_path = os.path.join(output_dir, f"{CONFIG['prompts']}_result.mp4")
        
        # 获取第一张图像的尺寸
        first_image = cv2.imread(image_paths[0])
        height, width, _ = first_image.shape
        
        # 设置视频参数
        original_fps = vr.get_avg_fps()  # 获取原始视频帧率
        fps = max(1, original_fps // vis_frame_stride)  # 计算输出帧率
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
        video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        print(f"开始合成视频: {video_output_path} (帧率: {fps}fps)")
        
        # 逐帧添加图像到视频
        for image_path in tqdm(image_paths, desc="合成视频"):
            img = cv2.imread(image_path)
            if img is not None and img.shape[:2] == (height, width):
                video.write(img)
            else:
                print(f"警告: 跳过尺寸不匹配的图像 {image_path}")
        
        # 释放视频写入器
        video.release()
        print(f"视频合成完成: {video_output_path}")
    else:
        print("未找到分割结果图像，无法创建视频")    
    # 不需要关闭VideoReader，decord会自动管理资源
    # 但可以显式删除以释放内存
    del vr   


if __name__ == "__main__":
    main()