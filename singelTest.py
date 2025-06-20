import os
import time
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.VisRetNet import VisRetNet
from timm.models.vision_transformer import _cfg
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2


def random_perturb_point(point, center, min_shift, max_shift):
    """随机扰动点的位置，扰动值在[min_shift, max_shift]之间，向图像中心扰动"""
    # 计算点到中心的向量
    vector_to_center = center - point
    # 计算扰动值，扰动值在[min_shift, max_shift]之间
    shift = np.random.uniform(min_shift, max_shift)
    # 根据扰动值缩放向量
    perturbation = shift * vector_to_center / np.linalg.norm(vector_to_center)
    return point + perturbation

def generate_homography_matrix(src_points, dst_points):
    """生成H矩阵"""
    # 使用RANSAC方法和5像素的容错距离来提高鲁棒性
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H

def apply_homography(image, H):
    """将H矩阵应用于图像"""
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (width, height))
    return warped_image

def add_random_rotation(points, center, angle_range=(-180, 180)):
    """添加随机角度旋转"""
    angle = np.random.uniform(angle_range[0], angle_range[1])
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    rotated_points = []
    for point in points:
        rotated_point = np.dot(rotation_matrix, np.array([point[0], point[1], 1.0]))
        rotated_points.append(rotated_point[:2])
    return np.array(rotated_points, dtype=np.float32)


def crop_center(image, crop_size):
    """从图像中心截取给定大小的图像"""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    half_crop_size = crop_size // 2
    start_x = max(center_x - half_crop_size, 0)
    start_y = max(center_y - half_crop_size, 0)
    end_x = min(center_x + half_crop_size, width)
    end_y = min(center_y + half_crop_size, height)
    return image[start_y:end_y, start_x:end_x]


def transform(input_image,size):


    input_image = crop_center(input_image,400)

    # 定义图像的四个角点
    height, width = input_image.shape[:2]
    src_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

    # 计算图像中心点
    center = np.array([width / 2, height / 2], dtype=np.float32)

    # 设置最小和最大扰动值
    min_shift = 0.05 * height  # 最小扰动值
    max_shift = 0.22 * height  # 最大扰动值

    # 对角点进行随机扰动，向图像中心扰动
    dst_points = np.array([random_perturb_point(point, center, min_shift, max_shift) for point in src_points], dtype=np.float32)

    # 添加随机旋转
    dst_points = add_random_rotation(dst_points, center)

    # 生成H矩阵
    H = generate_homography_matrix(src_points, dst_points)

    # 应用H矩阵
    output_image = apply_homography(input_image, H)

    output_image = crop_center(output_image, size)

    return output_image


def accuracy(query_features, reference_features, topk=[1,5,10]):
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])

    # 归一化，然后计算相似矩阵
    query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
    reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
    similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

    topk_similar_images = np.argsort(-similarity, axis=1)[:, :10]


    return topk_similar_images
  


def input_transform_query(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])


def main():
    # 模型初始化
    model = VisRetNet(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()

    # 加载权重
    loc = 'cuda:{}'.format(0)
    checkpoint = torch.load("./query_net_weights.pth", map_location=loc)
    model.load_state_dict(checkpoint, strict=True)
    model = model.cuda()
    model.eval()

    
    # 地图数据库
    data = np.load("./satelliteVector/reference_features_with_names.npz")
    image_names = data['image_names']
    reference_features = data['reference_features']   
    
    
    with torch.no_grad():
        successTop1 = 0
        paths = os.listdir("val_dataset")
        sorted_paths = sorted(paths, key=lambda x: int(x.split('_')[1].split('.')[0]))
        for ind,path in enumerate(sorted_paths):
            imgpath = os.path.join("val_dataset",path)
            # 读取图像
            img_query = cv2.imread(imgpath)
            img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)

            # 随机旋转+裁剪
            # img_query = transform(img_query,grd_size[0])

            # 图像预处理
            grd_size = [224, 224]
            transform_query = input_transform_query(size=grd_size)
            img_query = Image.fromarray(img_query)
            img_query = transform_query(img_query)
            images = img_query.unsqueeze(0)

            
            # compute output
            images = images.cuda(0, non_blocking=True)
            query_embed = model(images)
            query_embed = query_embed.cpu().numpy()
            

            
            topk_similar_images = accuracy(query_embed, reference_features)
            result = topk_similar_images[0][0]

            if ind == result:
                successTop1 += 1
            
            # print(result,image_names[result])

    print(successTop1 / len(paths))



if __name__ == '__main__':
    main()