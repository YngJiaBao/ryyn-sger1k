from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import io
import base64

# 初始化FastAPI应用
app = FastAPI()

# 加载YOLO模型
from ultralytics import YOLO
# 加载损伤检测模型
model = YOLO('best1.pt')
# 加载车身部位识别模型
parts_model = YOLO('best.pt')

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启用YOLO模型加载
model_loaded = True
print("YOLO model loaded successfully")

# 计算IOU（交并比）
def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比
    box格式: [x1, y1, x2, y2]
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IOU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou



# 车身部位映射
BODY_PARTS = {
    'Roof_outer_panel': '车顶外板',
    'Outer_mirror(right)': '外后视镜(右)',
    'Outer_mirror(left)': '外后视镜(左)',
    'Mirror_cover(right)': '后视镜盖(右)',
    'Mirror_cover(left)': '后视镜盖(左)',
    'Bottom_edge(right)': '底边(右)',
    'Bottom_edge(left)': '底边(左)',
    'Steel_ring': '钢圈',
    'Baggage_cover': '行李盖',
    'Rear_bumper_electric_eye': '后保险杠电眼',
    'Rear_bumper_skin': '后保险杠皮',
    'Rear_bumper_decorative_light(right)': '后保险杠装饰灯(右)',
    'Rear_bumper_decorative_light(left)': '后保险杠装饰灯(左)',
    'Rear_window_glass': '后窗玻璃',
    'Rear_door_glass(right)': '后门玻璃(右)',
    'Rear_door_glass(left)': '后门玻璃(左)',
    'Back_door_shell(right)': '后门壳(右)',
    'Back_door_shell(left)': '后门壳(左)',
    'Rear_door_trim(right)': '后门饰板(右)',
    'Rear_door_trim(left)': '后门饰板(左)',
    'Back_door_handle(right)': '后门把手(右)',
    'Back_door_handle(left)': '后门把手(左)',
    'Rear_fender(right)': '后翼子板(右)',
    'Rear_fender(left)': '后翼子板(左)',
    'Rear_fender_wheel_eyebrow(right)': '后翼子板轮眉(右)',
    'Rear_fender_wheel_eyebrow(left)': '后翼子板轮眉(左)',
    'liftgate_glass': '尾门玻璃',
    'liftgate_shell': '尾门壳',
    'Tire': '轮胎',
    'Inner_tail_light(right)': '内尾灯(右)',
    'Inner_tail_light(left)': '内尾灯(左)',
    'Front_bumper_skin': '前保险杠皮',
    'Front_bumper_lower_grille': '前保险杠下格栅',
    'Head_lamp(right)': '前大灯(右)',
    'Head_lamp(left)': '前大灯(左)',
    'Front_window_glass': '前窗玻璃',
    'Front_door_glass(right)': '前门玻璃(右)',
    'Front_door_glass(left)': '前门玻璃(左)',
    'Car_right_door': '右侧车门',
    'Car_left_door': '左侧车门',
    'Front_door_trim(right)': '前门饰板(右)',
    'Front_door_trim(left)': '前门饰板(左)',
    'Front_door_handle(right)': '前门把手(右)',
    'Front_door_handle(left)': '前门把手(左)',
    'Fog_lamp(right)': '雾灯(右)',
    'Fog_lamp(left)': '雾灯(左)',
    'Front_fender(right)': '前翼子板(右)',
    'Front_fender(left)': '前翼子板(左)',
    'Front_fender_wheel_eyebrow(right)': '前翼子板轮眉(右)',
    'Front_fender_wheel_eyebrow(left)': '前翼子板轮眉(左)',
    'Exterior_tail_light(right)': '外尾灯(右)',
    'Exterior_tail_light(left)': '外尾灯(左)',
    'Tail_light(right)': '尾灯(右)',
    'Tail_light(left)': '尾灯(左)',
    'Fuel_tank_cap': '油箱盖',
    'Grille': '格栅',
    'Grille_logo': '格栅标志',
    'Engine_cover': '发动机盖',
    'License_plate': '车牌'
}

# 损伤类型映射
DAMAGE_TYPES = {
    'scratch': '划痕',
    'dent': '凹陷',
    'crack': '裂纹',
    'broken': '破碎',
    'Glass breakage': '玻璃破碎',
    'Severe_deformation': '严重变形',
    'Dislocation': '错位',
    'Glass_breakage': '玻璃破碎',
    'Glass_crack': '玻璃裂纹',
    'Missing': '缺失',
    'Mild_deformation': '轻微变形',
    'Scratch': '划痕',
    'Scratches': '多处划痕',
    'Tearing': '撕裂'
}

# 严重程度映射
SEVERITY_LEVELS = {
    'minor': '轻微',
    'moderate': '中度',
    'severe': '严重'
}

@app.post("/api/recognizeDamage")
async def recognize_damage(file: UploadFile = File(...)):
    try:
        # 读取上传的图片
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 使用损伤检测模型进行检测
        damage_results = model(img_cv)
        
        # 使用车身部位识别模型进行检测
        parts_results = parts_model(img_cv)
        
        # 提取车身部位信息
        parts = []
        for part_result in parts_results[0].boxes:
            part_class_id = int(part_result.cls[0])
            part_name = parts_model.names[part_class_id]
            # 转换为中文
            part_name_cn = BODY_PARTS.get(part_name, part_name)
            part_confidence = float(part_result.conf[0])
            part_box = part_result.xyxy[0].tolist()
            
            parts.append({
                "name": part_name_cn,
                "confidence": part_confidence,
                "box": part_box
            })
        
        # 处理损伤检测结果
        damages = []
        for damage_result in damage_results[0].boxes:
            class_id = int(damage_result.cls[0])
            class_name = model.names[class_id]
            # 转换为中文
            class_name_cn = DAMAGE_TYPES.get(class_name, class_name)
            confidence = float(damage_result.conf[0])
            damage_box = damage_result.xyxy[0].tolist()
            
            # 计算损伤严重程度（基于相对面积比）
            # 计算损伤框面积
            damage_box_xywh = damage_result.xywh[0]
            damage_area = (damage_box_xywh[2] * damage_box_xywh[3]).item()
            
            # 计算车身部件框面积
            part_area = 0
            best_iou = 0
            best_part = "车辆"
            for part in parts:
                iou = calculate_iou(damage_box, part["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_part = part["name"]
                    # 计算车身部件框面积
                    part_box = part["box"]
                    part_area = (part_box[2] - part_box[0]) * (part_box[3] - part_box[1])
            
            # 计算相对面积比
            if part_area > 0:
                r = damage_area / part_area
            else:
                # 如果没有找到对应的车身部件，使用置信度来评估
                if confidence > 0.8:
                    r = 0.4  # 假设为严重损伤
                elif confidence > 0.6:
                    r = 0.2  # 假设为中度损伤
                else:
                    r = 0.05  # 假设为轻微损伤
            
            # 按比值分级
            if r < 0.1:
                severity = "minor"  # 轻微损伤
            elif r < 0.3:
                severity = "moderate"  # 中度损伤
            else:
                severity = "severe"  # 严重损伤
            
            damages.append({
                "type": class_name_cn,
                "location": best_part,
                "severity": severity,
                "confidence": confidence,
                "box": damage_box
            })
        
        # 转换回PIL格式
        annotated_img = damage_results[0].plot()
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        
        # 转换为base64编码
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "data": {
                "damages": damages,
                "imageUrl": f"data:image/jpeg;base64,{img_str}",
                "isOffline": False
            }
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

import aiohttp
import json

# 阿里云通义千问API配置
QIANWEN_API_KEY = "sk-d52ba4f542cd4074bc6b4e58af470627"
# 使用支持图片输入的API端点
QIANWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

from fastapi import Body

@app.post("/api/consultLLM")
async def consult_llm(request_data: dict = Body(...)):
    try:
        # 获取损伤数据
        damageData = request_data.get('damageData', {})
        damages = damageData.get('damages', [])
        
        # 构造提示词
        prompt = f"""
        你现在是【AI智能汽车定损专属解析顾问】，对接YOLO目标检测预处理后的汽车损伤标注图片开展工作；
        你的核心能力与约束：
        1. 优先读取损伤信息中的【损伤坐标、损伤类别、检测置信度、受损部件位置】，结合汽车定损行业标准、主流车型维保市场价、车险通用规则综合分析；
        2. 只做客观AI辅助判定，所有结论标注：仅为智能识别参考，最终定损金额、理赔结论以保险公司线下专员核验/官方定损报告为准；
        3. 语言通俗口语化，面向C端车主用户，少用硬核算法术语，分点清晰、不绕弯，共情用户车辆受损心情；
        4. 禁止引导骗保、违规定损、虚假维修建议，识别模糊/遮挡严重/YOLO低置信度损伤时主动提示风险并建议人工复核；
        5. 输出固定结构：损伤清单识别→严重等级评级→维保方案建议→费用区间参考→拍照/补材优化建议→理赔适配提醒。
        
        损伤信息：
        {json.dumps(damages, ensure_ascii=False, indent=2)}
        
        请基于上述损伤信息进行分析，提供详细的定损报告。
        """
        
        # 调用阿里云通义千问API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {QIANWEN_API_KEY}"
        }
        
        # 构建请求体，只传递损伤信息
        payload = {
            "model": "qwen-plus",  # 使用兼容模式中存在的模型
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        print("准备调用千问API")
        print("URL:", QIANWEN_API_URL)
        print("Headers:", headers)
        print("Payload:", payload)
        
        # 调用千问API
        try:
            # 使用异步aiohttp请求
            async with aiohttp.ClientSession() as session:
                async with session.post(QIANWEN_API_URL, headers=headers, json=payload, timeout=60) as response:
                    print("响应状态码:", response.status)
                    response_text = await response.text()
                    print("响应内容:", response_text)
                    response.raise_for_status()
                    result = await response.json()
                    print("解析后的响应:", result)
        except aiohttp.ClientError as e:
            print(f"HTTP错误: {str(e)}")
            # 抛出错误，不使用模拟数据
            raise
        except Exception as e:
            print(f"调用千问API失败: {str(e)}")
            # 抛出错误，不使用模拟数据
            raise
        
        # 解析阿里云通义千问返回的结果
        llm_response = result["choices"][0]["message"]["content"]
        print("千问模型响应:", llm_response)
        
        # 提取结构化信息
        damageList = ""
        severityLevel = ""
        maintenancePlan = ""
        costRange = ""
        photoAdvice = ""
        claimReminder = ""
        
        # 简单解析（实际项目中可能需要更复杂的解析逻辑）
        lines = llm_response.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if "损伤清单识别" in line:
                current_section = "damageList"
            elif "严重等级评级" in line:
                current_section = "severityLevel"
            elif "维保方案建议" in line:
                current_section = "maintenancePlan"
            elif "费用区间参考" in line:
                current_section = "costRange"
            elif "拍照/补材优化建议" in line:
                current_section = "photoAdvice"
            elif "理赔适配提醒" in line:
                current_section = "claimReminder"
            elif line and current_section == "damageList":
                damageList += line + "\n"
            elif line and current_section == "severityLevel":
                severityLevel += line + "\n"
            elif line and current_section == "maintenancePlan":
                maintenancePlan += line + "\n"
            elif line and current_section == "costRange":
                costRange += line + "\n"
            elif line and current_section == "photoAdvice":
                photoAdvice += line + "\n"
            elif line and current_section == "claimReminder":
                claimReminder += line + "\n"
        
        # 如果解析失败，使用默认值
        if not damageList:
            damageList = "未检测到车辆损伤，无需维修。"
        if not severityLevel:
            severityLevel = "无损伤"
        if not maintenancePlan:
            maintenancePlan = "根据损伤情况，建议及时进行维修。"
        if not costRange:
            costRange = "请咨询专业维修人员获取准确报价。"
        if not photoAdvice:
            photoAdvice = "建议在光线充足的环境下拍摄车辆损伤照片，确保损伤细节清晰可见。"
        if not claimReminder:
            claimReminder = "仅为智能识别参考，最终定损金额、理赔结论以保险公司线下专员核验/官方定损报告为准。"
        
        # 构建响应数据
        response_data = {
            "success": True,
            "data": {
                "damageList": damageList.strip(),
                "severityLevel": severityLevel.strip(),
                "maintenancePlan": maintenancePlan.strip(),
                "costRange": costRange.strip(),
                "photoAdvice": photoAdvice.strip(),
                "claimReminder": claimReminder.strip(),
                "recommendedActions": ["联系专业维修人员", "获取维修报价", "安排维修时间"]
            }
        }
        
        print("返回响应:", response_data)
        return response_data
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)