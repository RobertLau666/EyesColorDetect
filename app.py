import cv2
import numpy as np
import time
import os
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from tqdm import tqdm
import csv
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import dlib
import pandas as pd


def url_prefix_map(url):
    url = url.strip().replace('ali-us-sync-image.oss-us-east-1.aliyuncs.com', 'ali-bj-sync-image.oss-accelerate.aliyuncs.com')
    return url

def get_img_name_from_url(url):
    img_name = url.replace('?x-oss-process=image/resize,w_1080/format,webp', '').split('/')[-1]
    return img_name

def load_image(url):
    img = None
    timeout_num = 0
    timeout_max_try_num = 3
    load_img_stuck_wait_time = 5
    while True:
        try:
            response = requests.get(url, timeout=30)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            break
        except Exception as e:
            print(f"load image {url} stuck, try to load again after {load_img_stuck_wait_time} seconds, e: {str(e)}")
            time.sleep(load_img_stuck_wait_time)
            timeout_num += 1
            if timeout_num == timeout_max_try_num:
                print(f"load image {url} stuck, the max try num {timeout_max_try_num} has been reached, check the network and try again")
                break
    return img

# 生成html文件的函数
def generate_html_from_csv(csv_path, output_html_path):
    df = pd.read_csv(csv_path)
    
    # 计算预测准确率
    total_predictions = len(df)
    correct_predictions = (df['true_label'] == df['predict_label']).sum()
    accuracy = correct_predictions / total_predictions * 100  # 转换为百分比
    
    # 生成混淆矩阵
    confusion_matrix = pd.crosstab(df['true_label'], df['predict_label'], rownames=['Actual'], colnames=['Predicted'], margins=True)

    # 添加 img_name1 和 img_name2 列
    df['img_name1'] = df['url'].apply(lambda x: get_img_name_from_url(x))
    df['img_name2'] = df['input_url'].apply(lambda x: get_img_name_from_url(x))

    # 将 `url` 和 `input_url` 列替换为 HTML img 标签
    df['url'] = df['url'].apply(lambda x: f'<img src="{x}" width="100">')
    df['input_url'] = df['input_url'].apply(lambda x: f'<img src="{x}" width="100">')
    
    # 将准确率、混淆矩阵和原始数据转换为HTML
    html_content = (
        f"<h2>Overall Accuracy: {accuracy:.2f}%</h2>" +  # 显示准确率
        "<h2>Confusion Matrix</h2>" +
        confusion_matrix.to_html() +
        "<h2>Data Table</h2>" +
        df.to_html(escape=False, index=False)  # escape=False 允许 HTML 标签生效
    )
    
    # 保存为HTML文件
    with open(output_html_path, 'w') as f:
        f.write(html_content)


class EyeColorDetector:
    def __init__(self,eye_frame_scale=1, eye_color_same_score_threshold=0.313):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.eye_frame_scale = eye_frame_scale
        self.eye_color_same_score_threshold = eye_color_same_score_threshold

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def compute_histogram_similarity(self, hist1, hist2, method="bhattacharyya"):
        """计算两个直方图的相似度，返回值越接近1表示越相似"""
        hist1 = hist1 / hist1.sum()  # 归一化
        hist2 = hist2 / hist2.sum()
        
        if method == "bhattacharyya":
            distance = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            return 1 - distance  # 使得越相似，值越接近 1
        
        elif method == "kl_divergence":
            distance = entropy(hist1 + 1e-10, hist2 + 1e-10)  # 避免 log(0) 问题
            return 1 / (1 + distance)  # 使得越相似，值越接近 1
        
        elif method == "cosine":
            return 1 - cosine(hist1, hist2)  # 余弦相似度本身符合要求
        
        else:
            raise ValueError("Unsupported similarity method")

    def get_eye_hue_hist(self, image, eye_region, scale=0.4, bins=180):
        """计算眼睛区域的 H 通道直方图"""
        h, w = eye_region.shape[:2]
        center = (w // 2, h // 2)  
        radius = int(min(w, h) * scale)
        
        mask_circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_circle, center, radius, 255, -1)
        
        hsv_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        h_channel = hsv_region[:, :, 0]  # 仅提取 H 通道

        hist = cv2.calcHist([h_channel], [0], mask_circle, [bins], [0, 180])
        return hist.flatten()

    def compare_eye_hue_similarity(self, image1, image2, eye1, eye2, scale=0.4, method="bhattacharyya"):
        """计算两个眼睛区域的 H 颜色分布相似度"""
        hist1 = self.get_eye_hue_hist(image1, eye1, scale)
        hist2 = self.get_eye_hue_hist(image2, eye2, scale)
        
        similarity = self.compute_histogram_similarity(hist1, hist2, method)
        return similarity

    # 1. 不太准 检测到了多个眼睛
    def detect_eyes_haarcascade(self, image):
        """检测眼睛区域"""
        if image is None:
            print("图片未正确加载")
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        eyes = [eye.astype(int).tolist() for eye in eyes]
        # print('初始的 eyes: ', eyes)
        return eyes
    
    # 计算眼睛区域边界框
    def get_eye_bbox(self, points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return [min(x), min(y), max(x)-min(x), max(y)-min(y)]
        
    # 3. dlib
    def detect_eyes_dlib(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        print("faces: ", faces)
        if len(faces) == 0:
            eyes = []
        else:
            # 获取68个面部关键点
            landmarks = self.predictor(gray, faces[0])
            print("landmarks: ", landmarks)
            
            # 左眼关键点(36-41)
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            print("left_eye_points: ", left_eye_points)
            # 右眼关键点(42-47)
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            print("right_eye_points: ", right_eye_points)
            
            eyes = [self.get_eye_bbox(left_eye_points), self.get_eye_bbox(right_eye_points)]
        return eyes
    
    # 2. dlib + haarcascade
    def detect_eyes(self, image):
        """使用dlib关键点检测眼睛区域"""
        if image is None:
            print("图片未正确加载")
            return []
        
        # 先用 dlib检测眼睛
        eyes = self.detect_eyes_dlib(image)
        # 如果没有检测到眼睛，则使用 haarcascade 检测
        if eyes == []:
            print("换个方法检测眼睛")
            eyes = self.detect_eyes_haarcascade(image)
        print("eyes0: ", eyes)
        
        return eyes


    def get_valid_eyes(self, image, eyes): 
        valid_eyes = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in eyes:
            eye_region = gray[y:y+h, x:x+w]

            # 计算黑色和白色像素的数量
            black_pixels = np.sum(eye_region < 50)
            white_pixels = np.sum(eye_region > 200)

            # 如果同时存在足够数量的黑色和白色像素，则认为是有效的眼睛区域
            if black_pixels > 0 and white_pixels > 0:
                valid_eyes.append([x, y, w, h])
        # print('有效的 eyes: ', eyes)
        return valid_eyes

    def resize_eyes(self, eyes):
        # 缩小眼睛区域框为原来的一半，中心保持不变
        
        if self.eye_frame_scale != 1:
            # print("eye_frame_scale: ", eye_frame_scale)
            resized_eyes = []
            for (x, y, w, h) in eyes:
                # 计算缩小后的宽度和高度
                new_w = w // self.eye_frame_scale
                new_h = h // self.eye_frame_scale
                
                # 计算原始区域的中心点
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 计算缩小后的区域的左上角坐标
                new_x = center_x - new_w // self.eye_frame_scale
                new_y = center_y - new_h // self.eye_frame_scale
                
                # 将缩小后的区域添加到结果中
                resized_eyes.append([int(new_x), int(new_y), int(new_w), int(new_h)])
            eyes = resized_eyes
        
        # print('缩小的 eyes: ', eyes)
        return eyes

    def draw_eyes(self, image, eyes):
        """在图片上绘制眼睛区域的框"""
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 画绿色框
        return image

    def save_image_with_eyes(self, image, filename):
        """保存带眼睛框的图片"""
        name, ext = filename.rsplit('.', 1)  # 分离文件名和扩展名
        # print("image_dir: ", image_dir)
        # print("name: ", name)
        new_filename = f"{name}_eyes.{ext}"  # 添加 _eyes 后缀
        cv2.imwrite(new_filename, image)
        # print(f"xx图片已保存为: {new_filename}")

    def get_average_eye_color(self, image, eyes, scale=0.4, tolerance=50, color_threshold=50):
        """计算眼睛区域的平均颜色，使用眼镜框中心为圆心，边长的scale倍长为半径，
        去除偏白色、偏黑色区域，并去掉与上下两部分颜色相近的区域，计算圆内剩余区域的平均颜色"""
        
        eye_colors = []
        # 创建与原图尺寸相同的空mask，单通道（灰度图）
        composite_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # # 定义偏白色和偏黑色的范围（在 HSV 颜色空间中）
        # lower_white = np.array([0, 0, 200])   # 亮度较高，认为是白色
        # upper_white = np.array([180, 30, 255])
        
        # lower_black = np.array([0, 0, 0])       # 亮度较低，认为是黑色
        # upper_black = np.array([180, 255, 50])

        # # 定义偏白色和偏黑色的范围（在 HSV 颜色空间中）
        # lower_white = np.array([0, 0, 255-color_threshold])   # 亮度较高，认为是白色
        # upper_white = np.array([180, 30, 255])
        
        # lower_black = np.array([0, 0, 0])       # 亮度较低，认为是黑色
        # upper_black = np.array([180, 255, color_threshold])
        
        # 定义偏白色和偏黑色的范围（在 HSV 颜色空间中）
        lower_white = np.array([0, 0, 200])   # 亮度较高，认为是白色
        upper_white = np.array([180, 30, 255])
        
        lower_black = np.array([0, 0, 0])       # 亮度较低，认为是黑色
        upper_black = np.array([180, 255, 0])

        for (x, y, w, h) in eyes:
            # 提取眼睛区域
            eye_region = image[y:y+h, x:x+w]
            # 将眼睛区域从 BGR 转换为 HSV 颜色空间
            hsv_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
            
            # 计算眼镜框中心和半径
            center = (w // 2, h // 2)  # 眼睛区域的相对中心
            radius = int(min(w, h) * scale)  # 半径为眼睛区域最小边长的 scale 倍
            
            # 创建圆形掩码（相对于当前眼睛区域的坐标）
            mask_circle = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_circle, (center[0], center[1]), radius, 255, -1)  # 在眼睛区域内创建圆形掩码
            
            # 创建掩码，排除偏白色和偏黑色的像素
            mask_white = cv2.inRange(hsv_region, lower_white, upper_white)
            mask_black = cv2.inRange(hsv_region, lower_black, upper_black)
            mask = cv2.bitwise_not(cv2.bitwise_or(mask_white, mask_black))
            
            # 计算上下两部分的颜色
            upper_half = hsv_region[:h//2, :]
            lower_half = hsv_region[h//2:, :]
            
            # 上半部分和下半部分的平均颜色
            avg_upper = np.mean(upper_half, axis=(0, 1))
            avg_lower = np.mean(lower_half, axis=(0, 1))
            
            # 创建上下部分颜色的掩码
            mask_upper = cv2.inRange(hsv_region, np.maximum(avg_upper - tolerance, [0, 0, 0]), np.minimum(avg_upper + tolerance, [179, 255, 255]))
            mask_lower = cv2.inRange(hsv_region, np.maximum(avg_lower - tolerance, [0, 0, 0]), np.minimum(avg_lower + tolerance, [179, 255, 255]))
            
            # 合并上下部分颜色掩码
            mask_vertical = cv2.bitwise_or(mask_upper, mask_lower)
            
            # 将圆形掩码和颜色范围掩码合并
            final_mask = cv2.bitwise_and(mask, mask_circle)
            final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(mask_vertical))
            
            # 将当前眼睛区域的最终mask合并到整体mask中，保持在原图中的位置
            roi = composite_mask[y:y+h, x:x+w]
            composite_mask[y:y+h, x:x+w] = cv2.bitwise_or(roi, final_mask)
            
            # 计算过滤后的平均颜色（仅考虑色调 H 和饱和度 S，忽略亮度 V）
            non_zero_pixels = hsv_region[final_mask != 0]
            if non_zero_pixels.size > 0:
                average_color = np.mean(non_zero_pixels, axis=0)
                eye_colors.append(average_color[:2])  # 只取色调 H 和饱和度 S
        
        return eye_colors, composite_mask

    def color_difference(self, color1, color2):
        """计算颜色差异（欧几里得距离）"""
        return np.linalg.norm(color1 - color2)

    def __call__(self, image1_path, image2_path):
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # 0. 检查图片是否加载成功
        if image1 is None:
            print(f"无法加载 {image1_path}，请检查文件路径")
        if image2 is None:
            print(f"无法加载 {image2_path}，请检查文件路径")

        # 1. 检测眼睛区域
        print("11")
        eyes1 = self.detect_eyes(image1)
        print("22")
        eyes2 = self.detect_eyes(image2)
        print("1")
        print("eyes1: ", eyes1)
        print("eyes2: ", eyes2)
        
        # # 2. 筛选有效眼睛区域
        # eyes1 = get_valid_eyes(image1, eyes1)
        # eyes2 = get_valid_eyes(image2, eyes2)
        # print("2")
        # print("eyes1: ", eyes1)
        # print("eyes2: ", eyes2)

        # 3. 缩放眼睛区域
        eyes1 = self.resize_eyes(eyes1)
        eyes2 = self.resize_eyes(eyes2)
        print("3")
        print("eyes1: ", eyes1)
        print("eyes2: ", eyes2)

        # 4. 在图片上绘制眼睛框
        if len(eyes1) > 0:
            image1_with_eyes = self.draw_eyes(image1.copy(), eyes1)
            self.save_image_with_eyes(image1_with_eyes, image1_path)
        if len(eyes2) > 0:
            image2_with_eyes = self.draw_eyes(image2.copy(), eyes2)
            self.save_image_with_eyes(image2_with_eyes, image2_path)

        # 5. 提取眼睛区域的平均颜色
        eye_colors1, composite_mask1 = self.get_average_eye_color(image1, eyes1)
        eye_colors2, composite_mask2 = self.get_average_eye_color(image2, eyes2)
        cv2.imwrite(f"{image_dir}/{image1_path}_mask.png", composite_mask1)
        cv2.imwrite(f"{image_dir}/{image2_path}_mask.png", composite_mask2)

        # # 6. 计算色差（不用这种计算方式了）
        # eye_color_diff_score = -1
        # is_same = False
        # if eye_colors1 and eye_colors2:
        #     color1 = eye_colors1[0]  # 取第一只眼睛的平均颜色
        #     color2 = eye_colors2[0]  # 取第一只眼睛的平均颜色
        #     eye_color_diff_score = color_difference(color1, color2)
        #     if eye_color_diff_score < eye_color_diff_threshold:  # 阈值可根据实际情况调整
        #         is_same = True
        #     else:
        #         is_same = False
        # else:
        #     print("未检测到眼睛")
        print("4")
        if len(eye_colors1) == 0 or len(eye_colors2) == 0:
            print("至少有一张图未检测到眼睛，无法计算，直接返回默认无效值")
            return -1, False

        print("eyes1: ", eyes1)
        x, y, w, h = eyes1[0]
        eye1 = image1[y:y+h, x:x+w]
        print("eyes2: ", eyes2)
        x, y, w, h = eyes2[0]
        eye2 = image2[y:y+h, x:x+w]

        # 保存眼睛图像
        cv2.imwrite(os.path.join(image_dir, f"{os.path.splitext(os.path.basename(image1_path))[0]}_eye0.jpg"), cv2.cvtColor(eye1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(image_dir, f"{os.path.splitext(os.path.basename(image2_path))[0]}_eye0.jpg"), cv2.cvtColor(eye2, cv2.COLOR_RGB2BGR))

        eye_color_same_score = self.compare_eye_hue_similarity(image1, image2, eye1, eye2, method="bhattacharyya")
        # print("H 颜色分布相似度:", eye_color_same_score)

        if eye_color_same_score > eye_color_same_score_threshold:
            is_same = True
        else:
            is_same = False

        return eye_color_same_score, is_same


if __name__ == "__main__":
    eye_frame_scale = 1#2
    # eye_color_diff_threshold = 30
    eye_color_same_score_threshold = 0.313
    image_dir = "images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    eye_color_detector = EyeColorDetector(eye_frame_scale=eye_frame_scale, eye_color_same_score_threshold=eye_color_same_score_threshold)
    
    eye_color_same_scores = []
    is_sames = []

    # # 1 image_path
    # image_names = [
    #     ['1_0.png', '1_2.png'],
    #     ['1_1.png', '1_2.png'],
    #     ['2_0.png', '2_2.png'],
    #     ['2_1.png', '2_2.png'],
    #     ['3_0.png', '3_2.png'],
    #     ['3_1.png', '3_2.png'],

    #     ['1_2.png', '2_2.png'],
    #     ['2_1.png', '2_1.png'],
    #     ['3_2.png', '3_2.png'],
    #     ['3_1.png', '3_2.png'],
    # ]
    # for image_name in tqdm(image_names):
    #     print("\n\nimage_name: ", image_name)
    #     image1_path, image2_path = os.path.join(image_dir, image_name[0]), os.path.join(image_dir, image_name[1])

    #     start_time = time.time()
    #     eye_color_same_score, is_same = pipeline(image1_path, image2_path, eye_frame_scale) # , eye_color_diff_threshold
    #     end_time = time.time()
    #     cost_time = end_time - start_time

    #     print(f"\nimage_name: {image_name}")
    #     print(f"眼睛区域平均颜色相似度: {eye_color_same_score}")
    #     print(f"眼睛区域平均颜色一致吗: {is_same}")
    #     print(f"耗时（s）: {cost_time}")
    #     eye_color_same_scores.append(eye_color_same_score)
    #     is_sames.append(is_same)


    # 2 image_url
    input_csv_path = 'test_eyes_color.csv'
    df = pd.read_csv(input_csv_path)
    new_csv_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if index == 10:
            break
        url, input_url = url_prefix_map(row[0]), url_prefix_map(row[1])
        img1_name, img2_name = get_img_name_from_url(url), get_img_name_from_url(input_url)
        img1_save_path, img2_save_path = os.path.join(image_dir, img1_name), os.path.join(image_dir, img2_name)
        print("index: ", index, img1_save_path, img2_save_path)

        if not os.path.exists(img1_save_path):
            img1 = load_image(url)
            if img1 is not None:
                img1.save(img1_save_path)
        if not os.path.exists(img2_save_path):
            img2 = load_image(input_url)
            if img2 is not None:
                img2.save(img2_save_path)

        start_time = time.time()
        try:
            eye_color_same_score, is_same = eye_color_detector(img1_save_path, img2_save_path)
        except Exception as e:
            print(f"Error processing {img1_save_path} and {img2_save_path}: {e}")
            eye_color_same_score, is_same = -1, False
            continue
        end_time = time.time()
        cost_time = end_time - start_time

        # print(f"\nimage_name: {image_name}")
        print(f"眼睛区域平均颜色相似度: {eye_color_same_score}")
        print(f"眼睛区域平均颜色一致吗: {is_same}")
        predict_label = '1' if is_same else '0'
        print(f"耗时（s）: {cost_time}")
        print(f"\n\n\n\n\n")
        eye_color_same_scores.append(eye_color_same_score)
        is_sames.append(is_same)

        new_csv_data.append(row.tolist() + [predict_label, eye_color_same_score, cost_time])

    print("eye_color_same_scores: ", eye_color_same_scores)
    print("is_sames: ", is_sames)

    # 保存新的CSV文件
    new_csv_data.insert(0, df.columns.tolist() + ['predict_label', 'eye_color_same_score', 'cost_time'])  # 添加标题行
    output_csv_path = os.path.join("outputs", input_csv_path.replace('.csv', '_output.csv'))
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_csv_data)
    
    # 生成HTML文件
    output_html_path = output_csv_path.replace('.csv', '.html')
    generate_html_from_csv(output_csv_path, output_html_path)
    print(f"生成的HTML文件路径: {output_html_path}")

# python app.py > outputs/test_eyes_color_output.log 2>&1 &