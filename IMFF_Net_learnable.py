import torch
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # PyTorch优化器
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
from torchvision import models, transforms, datasets  # Torchvision图像处理工具
from torch.utils.data import Subset  # 数据集子集处理
from PIL import Image  # 图像处理库
import os  # 操作系统接口
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
import torch.nn.functional as F  # PyTorch函数式API
from sklearn.metrics import confusion_matrix  # 混淆矩阵计算
import matplotlib.pyplot as plt  # 绘图库
import seaborn as sns  # 数据可视化
import numpy as np  # 数值计算库
from scipy import ndimage  # 图像处理函数
import pandas as pd  # 数据处理库
import time  # 时间模块
from sklearn.metrics import cohen_kappa_score  # Cohen's Kappa系数计算
from sklearn.manifold import TSNE  # t-SNE降维可视化

# ===== 全局超参数定义 =====
image_size = 224  # 输入图像尺寸
batch_size = 16   # 批处理大小
num_classes = 45  # 分类类别数
num_epochs = 30   # 训练轮数
learning_rate = 0.0002  # 初始学习率
min_lr = 1e-6     # 最小学习率

def get_log_kernel(size, sigma):
    """
    创建2D LoG(Laplacian of Gaussian)滤波器
    
    参数:
        size: 滤波器尺寸
        sigma: 高斯核的标准差
        
    返回:
        二维LoG滤波器核(tensor)
    """
    # 创建坐标网格
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    y = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(x, y)
    
    # 计算LoG核 (高斯二阶导)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (1 * sigma ** 2))
    kernel = kernel * (xx ** 2 + yy ** 2 - 1 * sigma ** 2) / (sigma ** 4)
    
    return torch.from_numpy(kernel.astype(np.float32))

class LogLayer(nn.Module):
    """
    LoG(高斯拉普拉斯)卷积层 - 用于边缘特征提取
    
    参数:
        channels: 输入通道数
        size: 滤波器尺寸
        sigma: 高斯标准差
    """
    def __init__(self, channels, size, sigma):
        super(LogLayer, self).__init__()
        # 创建不可训练的LoG卷积核
        self.weight = nn.Parameter(
            get_log_kernel(size, sigma).unsqueeze(0).unsqueeze(0), 
            requires_grad=False
        )
        self.groups = channels  # 分组卷积数（=输入通道数）
        
    def forward(self, x):
        # 将核扩展到所有输入通道
        weight = self.weight.repeat(self.groups, 1, 1, 1)
        # 执行分组卷积（各通道独立处理）
        return F.conv2d(x, weight, padding=1, groups=self.groups)

class MultiScaleBranchModule(nn.Module):
    """
    多尺度特征提取分支 - 同时使用不同大小的卷积核
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBranchModule, self).__init__()
        # 分支1：3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        # 分支2：5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # 分别处理两个分支
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)   
        # 沿通道维度拼接特征
        output = torch.cat([branch2_out, branch3_out], dim=1)
        return output

class ObjectFeatureBranch(nn.Module):
    """
    对象特征提取分支 - 包含多尺度处理和池化
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_classes: 类别数
    """
    def __init__(self, in_channels, out_channels, num_classes):
        super(ObjectFeatureBranch, self).__init__()
        # 多尺度特征提取
        self.multi_scale_branch = MultiScaleBranchModule(in_channels, out_channels)
        # 全局平均池化 - 减少空间维度
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.multi_scale_branch(x)  # 多尺度特征
        x = self.pool(x)               # 全局池化
        return x
    
class WeightedFusionLayer(nn.Module):
    """
    可学习加权特征融合层 - 用于融合两种特征
    
    特点:
        1. 使用可学习的融合权重
        2. 权重通过sigmoid约束在0-1之间
        3. 权重归一化确保总和为1
    """
    def __init__(self):
        super(WeightedFusionLayer, self).__init__()
        # 初始化两个可学习权重参数
        self.weight_out4 = nn.Parameter(torch.tensor(0.5))
        self.weight_out_log = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, out4, out_log):
        # 使用sigmoid将权重约束到(0,1)
        w_out4 = torch.sigmoid(self.weight_out4)
        w_out_log = torch.sigmoid(self.weight_out_log)
        
        # 权重归一化（确保和为1）
        total_weight = w_out4 + w_out_log
        w_out4 = w_out4 / total_weight
        w_out_log = w_out_log / total_weight
        
        # 加权融合两种特征
        return w_out4 * out4 + w_out_log * out_log
    
# 输出融合层 - 学习三个分类器的权重
class LearnableWeightsFusion(nn.Module):
    """
    可学习输出融合层 - 用于融合三个分类器输出
    
    参数:
        num_outputs: 分类器数量 (默认为3)
        random_init: 是否随机初始化 (False时使用预设权重初始化)
        
    预设权重比例:
        out_class: 0.57 (主分类器)
        out_custom: 0.16 (第一个辅助分类器)
        out_custom1: 0.27 (第二个辅助分类器)
    """
    def __init__(self, num_outputs=3, random_init=False):
        super(LearnableWeightsFusion, self).__init__()
        
        if random_init:
            # 随机初始化logits值
            initial_weights = torch.randn(num_outputs)
            std = 1.0 / np.sqrt(num_outputs)
            initial_weights = torch.randn(num_outputs) * std
        else:
            # 设置期望的初始权重 [0.57, 0.16, 0.27]
            desired_weights = torch.tensor([0.57, 0.16, 0.27])
            
            # 使用数值优化找到能产生期望权重的logits值
            logits = torch.zeros(num_outputs, requires_grad=True)
            optimizer = optim.LBFGS([logits], lr=1)
            
            # 优化循环
            for _ in range(100):  # 迭代优化
                def closure():
                    optimizer.zero_grad()
                    current_weights = F.softmax(logits, dim=0)
                    loss = ((current_weights - desired_weights) ** 2).sum()
                    loss.backward()
                    return loss
                
                optimizer.step(closure)
            
            initial_weights = logits.detach().clone()
            
        # 将logits初始化为参数
        self.logits = nn.Parameter(initial_weights)
        
    def forward(self, out_class, out_custom, out_custom1):
        # 使用softmax确保权重总和为1
        weights = F.softmax(self.logits, dim=0)
        
        # 将权重应用于三个分类器的输出
        out = weights[0] * out_class + weights[1] * out_custom + weights[2] * out_custom1
        
        # 返回加权输出和当前权重以进行监控
        return out, weights

def compute_entropy(feature_map, xs, xd, ys, yd):
    """
    计算特征图指定区域的熵 (信息熵)
    
    参数:
        feature_map: 特征图张量
        xs, xd: x方向起始和结束索引
        ys, yd: y方向起始和结束索引
        
    返回:
        区域熵值
    """
    # 1. 提取区域特征
    cropped_feature_map = feature_map[:, :, xs:xd, ys:yd]
    # 2. 展平特征
    flattened_feature_map = cropped_feature_map.view(-1)
    # 3. 计算直方图（256 bins）
    histogram = torch.histc(flattened_feature_map, bins=256, min=0, max=1)
    # 4. 计算概率分布
    prob_distribution = histogram / torch.sum(histogram)
    # 5. 计算熵值（避免log(0)）
    entropy = -torch.sum(prob_distribution * torch.log2(prob_distribution + 1e-9)) 
    return entropy

def compute_entropy_for_each_y(feature_map, xs, xd):
    """
    计算一行中每个y位置的熵
    
    用于EKLM算法的初始行分析
    
    参数:
        feature_map: 特征图张量
        xs, xd: x方向切片范围
    """
    entropies = []
    for ys in range(feature_map.shape[3]):
        entropy = compute_entropy(feature_map, xs, xd, ys, ys + 1)
        entropies.append(entropy)
    return torch.tensor(entropies)

def EKLM(A_hat, xs, T):
    """
    基于熵的关键区域选择算法 (Entropy-based Key Localization Method)
    
    步骤:
        1. 找到熵最大的初始位置
        2. 逐步扩展区域直到熵比例超过阈值T
        3. 优先扩展熵增量最大的方向
    
    参数:
        A_hat: 特征图 (B, C, H, W)
        xs: 起始x坐标
        T: 熵比例阈值
        
    返回:
        [xs, xd, ys, yd]: 选中的区域坐标
    """
    # 初始位置设置
    xs = 0
    # 计算第一行的熵分布
    entropies = compute_entropy_for_each_y(A_hat, xs, xs + 1)
    # 选择熵最大的y位置
    ys = torch.argmax(entropies)
    yd = ys + 1
    xd = xs + 1
    
    # 计算当前区域的熵占特征图总熵的比例
    total_entropy = compute_entropy(A_hat, 0, A_hat.shape[2], 0, A_hat.shape[3])
    Ts = compute_entropy(A_hat, xs, xd, ys, yd) / total_entropy
    
    # 区域扩展算法：当熵比例小于阈值时扩展区域
    while Ts < T:
        # 检查四个方向扩展的可能性和熵增量
        right_expand = (
            xd + 1 < A_hat.shape[2] and 
            compute_entropy(A_hat, xs, xd + 1, ys, yd) > 
            compute_entropy(A_hat, xs, xd, ys, yd) and 
            (ys == 0 or 
             compute_entropy(A_hat, xs, xd + 1, ys, yd) > 
             compute_entropy(A_hat, xs, xd, ys - 1, yd)) and 
            (yd + 1 == A_hat.shape[3] or 
             compute_entropy(A_hat, xs, xd + 1, ys, yd) > 
             compute_entropy(A_hat, xs, xd, ys, yd + 1))
        )
        
        up_expand = (
            ys - 1 >= 0 and 
            compute_entropy(A_hat, xs, xd, ys - 1, yd) > 
            compute_entropy(A_hat, xs, xd, ys, yd) and 
            (xd + 1 == A_hat.shape[2] or 
             compute_entropy(A_hat, xs, xd, ys - 1, yd) > 
             compute_entropy(A_hat, xs, xd + 1, ys, yd)) and 
            (yd + 1 == A_hat.shape[3] or 
             compute_entropy(A_hat, xs, xd, ys - 1, yd) > 
             compute_entropy(A_hat, xs, xd, ys, yd + 1))
        )
        
        down_expand = (
            yd + 1 < A_hat.shape[3] and 
            compute_entropy(A_hat, xs, xd, ys, yd + 1) > 
            compute_entropy(A_hat, xs, xd, ys, yd)
        )
        
        # 决策树：选择熵增量最大的方向
        if right_expand:
            xd = xd + 1
        elif up_expand:
            ys = ys - 1
        elif down_expand:
            yd = yd + 1
        else:  
            break  # 所有方向扩展都不满足条件时终止循环
            
        # 更新当前熵比例
        Ts = compute_entropy(A_hat, xs, xd, ys, yd) / total_entropy
    
    return [xs, xd, ys, yd]

class DDP(nn.Module):
    """
    深度可分离卷积模块 (Depthwise Separable Convolution)
    
    特点:
        1. 减少计算量和参数数量
        2. 保持特征提取能力
    
    参数:
        num_classes: 输出类别数
    """
    def __init__(self, num_classes):
        super(DDP, self).__init__()
        # 常规卷积 (降低通道数)
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 带负斜率的ReLU
        
        # 深度可分离卷积
        self.conv2_depthwise = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv2_pointwise = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # 特征处理流程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2_depthwise(x)   # 深度卷积
        x = self.conv2_pointwise(x)   # 逐点卷积
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avg_pool(x)          # 全局池化
        x = x.view(x.size(0), -1)     # 展平
        return x

class OmegaVisionNet(nn.Module):
    """
    OmegaVisionNet - 遥感图像分类模型
    
    主要特点:
        1. 基于DenseNet121主干
        2. 多分支特征提取
        3. LoG边缘特征提取
        4. EKLM关键区域选择
        5. 三分类器加权融合
    
    参数:
        num_classes: 输出类别数 (默认为45)
    """
    def __init__(self, num_classes=45):
        super(OmegaVisionNet, self).__init__()
        # 加载预训练的DenseNet121
        densenet121 = models.densenet121(pretrained=True)
        
        # === 自定义初始卷积层 (空洞卷积) ===
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, dilation=3)
        self.bn1 = nn.BatchNorm2d(256)
        
        # === 复用DenseNet的低级特征层 ===
        self.features_conv0 = densenet121.features.conv0
        self.features_norm0 = densenet121.features.norm0
        self.features_relu0 = densenet121.features.relu0
        self.features_pool0 = densenet121.features.pool0
        
        # === 对象特征分支 (不同层级) ===
        self.extra_feature_branch = ObjectFeatureBranch(64, 128, num_classes)
        self.extra_feature_branch1 = ObjectFeatureBranch(256, 128, num_classes)
        
        # === DenseNet主路径 ===
        self.features_denseblock1 = densenet121.features.denseblock1
        self.features_transition1 = densenet121.features.transition1
        self.features_denseblock2 = densenet121.features.denseblock2
        self.features_transition2 = densenet121.features.transition2
        self.features_denseblock3 = densenet121.features.denseblock3
        self.features_transition3 = densenet121.features.transition3
        self.features_denseblock4 = densenet121.features.denseblock4
        
        # === 特征融合组件 ===
        self.weighted_fusion = WeightedFusionLayer()  # 特征融合
        self.features_norm5 = densenet121.features.norm5
        self.classifier = nn.Linear(1792, num_classes)  # 主分类器
        
        # === 辅助特征处理 ===
        self.ddp = DDP(num_classes)  # 深度可分离卷积模块
        self.log_layer = LogLayer(channels=1024, size=3, sigma=0.5)  # LoG层
        self.fc = nn.Linear(64, num_classes)  # 辅助分类器
        
        # === 输出融合层 (学习三个分类器的权重) ===
        self.output_fusion = LearnableWeightsFusion(3)  # 权重：[0.57, 0.16, 0.27]
        
    def forward(self, x):
        # === 路径1: 自定义初始卷积路径 ===
        go = self.conv1(x)
        go = self.bn1(go)
        go = self.features_relu0(go)
        go = self.features_pool0(go)
        
        # === 路径2: 标准DenseNet前处理 ===
        out0 = self.features_conv0(x)
        out0 = self.features_norm0(out0)
        out0 = self.features_relu0(out0)
        out0 = self.features_pool0(out0)
        
        # === 对象特征提取 (早期特征) ===
        out_branch = self.extra_feature_branch(out0)
        
        # === DenseNet主通路 ===
        out = self.features_denseblock1(out0)
        out = self.features_transition1(out)
        out = self.features_denseblock2(out)
        out = self.features_transition2(out)
        
        # === 对象特征提取 (中期特征) ===
        out_branch1 = self.extra_feature_branch1(out)
        
        # === DenseNet深层特征 ===
        out = self.features_denseblock3(out)
        out = self.features_transition3(out)
        out4 = self.features_denseblock4(out)  # 用于EKLM的特征
        
        # === 特征归一化和池化 ===
        out5 = self.features_norm5(out4)
        out_pool = F.adaptive_avg_pool2d(out5, (1, 1))  # 全局池化
        out_pool = out_pool.view(out_pool.size(0), -1)  # 展平
        
        # === 处理分支1 ===
        go_pool = F.adaptive_avg_pool2d(go, (1, 1))
        go_pool = go_pool.view(go_pool.size(0), -1)
        
        # === LoG边缘特征处理 ===
        out_log = self.log_layer(out5)  # 应用LoG滤波
        out_log = self.weighted_fusion(out5, out_log)  # 融合原始特征和LoG特征
                             
        # === 基于熵的关键区域选择 (EKLM) ===
        xs = 0
        T = 0.4  # 熵比例阈值
        xs, xd, ys, yd = EKLM(out4.detach(), xs, T)  # 分离计算图避免梯度传播
        oute = out4[:, :, xs:xd, ys:yd]  # 裁剪关键区域特征
        
        # === DDP模块处理两种特征 ===
        out_custom = self.ddp(out_log)   # 处理原始特征+LoG
        out_custom1 = self.ddp(oute)     # 处理关键区域特征
        
        # === 辅助分类器 ===
        out_custom = self.fc(out_custom)
        out_custom1 = self.fc(out_custom1)
        
        # === 特征拼接 (主特征+分支特征) ===
        fused_out = torch.cat([
            out_pool, 
            out_branch1.view(out_branch1.size(0), -1), 
            out_branch.view(out_branch.size(0), -1), 
            go_pool
        ], dim=1)
        out_class = self.classifier(fused_out)  # 主分类器输出
        
        # === 融合三个分类器的输出 ===
        fused_out, fusion_weights = self.output_fusion(out_class, out_custom, out_custom1)
        
        return fused_out, fusion_weights

# === 数据增强: 随机亮度和对比度调整 ===
class RandomBrightnessContrast(transforms.RandomApply):
    """
    随机亮度和对比度增强
    概率p随机应用
    """
    def __init__(self, brightness=0.1, contrast=0.2, p=0.5):
        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        super().__init__(transforms=[transform], p=p)

# === 训练数据预处理 ===
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整尺寸
    transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转（50%概率）
    RandomBrightnessContrast(),                 # 自定义亮度对比度调整
    transforms.RandomCrop(image_size, padding=4, padding_mode='reflect'),  # 随机裁剪（带反射填充）
    transforms.ToTensor(),                       # 转为张量 [0,1]范围
    transforms.Normalize(                        # 标准化（ImageNet均值和标准差）
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

# === 测试数据预处理 ===
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整尺寸
    transforms.ToTensor(),                       # 转为张量
    transforms.Normalize(                        # 标准化
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

class AID20Dataset(Dataset):
    """
    遥感图像数据集加载器
    
    特点:
        1. 从目录结构自动分类
        2. 支持数据增强
        
    目录结构:
        data_dir/
            class_1/
                img1.jpg
                img2.jpg
                ...
            class_2/
                ...
            ...
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))  # 类别即目录名（排序确保一致）
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 类别到索引的映射
        self.filepaths = []  # 存储所有文件路径
        self.labels = []     # 存储对应标签
        
        # 遍历数据集目录并收集图像路径和标签
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):  # 图像文件检查
                    self.filepaths.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls])
                
    def __len__(self):
        """返回数据集大小"""
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert("RGB")  # 确保RGB格式（3通道）
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)  # 应用预处理
            
        return img, label
    
def get_accuracy(logit, target):
    """
    计算分类准确率
    
    参数:
        logit: 模型输出的logits [batch_size, num_classes]
        target: 真实标签 [batch_size]
        
    返回:
        准确率百分比
    """
    # 获取预测类别（最大logit的索引）
    _, preds = torch.max(logit, 1)
    # 计算正确预测数量
    corrects = (preds == target).sum().float()
    # 计算准确率百分比
    accuracy = 100.0 * corrects / target.size(0)
    return accuracy.item()

def main():
    """
    主训练函数
    
    流程:
        1. 数据准备
        2. 模型初始化
        3. 训练循环
        4. 验证评估
    """
    # === 数据准备 ===
    data_dir_train = '/root/autodl-tmp/Swin-Transformer-main/20'  # 训练集路径
    data_dir_test = '/root/autodl-tmp/Swin-Transformer-main/20test'  # 测试集路径
    
    # 创建数据集
    train_dataset = AID20Dataset(data_dir=data_dir_train, transform=train_transform)
    test_dataset = AID20Dataset(data_dir=data_dir_test, transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        shuffle=True,  # 训练时打乱数据
        pin_memory=True  # 固定内存加速传输
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        shuffle=False,  # 测试时不打乱
        pin_memory=True
    )
    
    # === 模型初始化 ===
    model = OmegaVisionNet(num_classes=num_classes)  # 创建模型
    
    # 加载预训练权重（使用非严格模式，允许部分不匹配）
    model_weights_path = '/root/autodl-tmp/NWPU20/model95.1.pth'
    model.load_state_dict(torch.load(model_weights_path), strict=False)
    
    # 设置优化器 (Adam)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4  # L2正则化
    )
    
    # 损失函数 - 交叉熵损失
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器 (余弦退火)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=20,       # 周期长度 (半周期数)
        eta_min=min_lr  # 最小学习率
    )
    
    # === GPU支持 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on: {device}")
    
    # === 训练循环 ===
    fusion_weights_history = []  # 记录权重变化（用于分析）
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # === 训练阶段 ===
        model.train()  # 设置为训练模式
        start_time = time.time()
        
        # 初始化统计量
        train_running_loss = 0.0
        train_acc = 0.0
        total_train_samples = 0
        epoch_fusion_weights = []  # 存储当前epoch的融合权重
        
        for i, (images, labels) in enumerate(train_loader):
            # 数据移动到设备
            images, labels = images.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播（获取分类结果和融合权重）
            outputs, fusion_weights = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            # 统计指标
            batch_size = labels.size(0)
            total_train_samples += batch_size
            train_running_loss += loss.item() * batch_size
            train_acc += get_accuracy(outputs, labels) * batch_size
            
            # 记录融合权重
            epoch_fusion_weights.append(fusion_weights.detach().cpu().numpy())
            
        # 计算epoch平均指标
        train_running_loss /= total_train_samples
        train_acc /= total_train_samples
        avg_fusion_weights = np.mean(epoch_fusion_weights, axis=0)
        fusion_weights_history.append(avg_fusion_weights)
        
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f}s")
        
        # 计算模型大小
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_params_mb = train_params * 4 / (1024 ** 2)  # 估算内存占用 (float32 = 4字节)
        print(f"Trainable Parameters: {train_params_mb:.2f} MB")
        
        # 输出当前融合权重
        print(f"Current fusion weights: {avg_fusion_weights}")
        
        # === 验证阶段 ===
        start_time = time.time()
        model.eval()  # 设置为评估模式
        test_acc = 0.0
        total_test_samples = 0
        all_preds = []    # 存储所有预测
        all_targets = []   # 存储所有真实标签
        class_correct = np.zeros(num_classes)  # 每个类别的正确预测数
        class_count = np.zeros(num_classes)    # 每个类别的总样本数
        
        # 禁用梯度计算
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播
                outputs, _ = model(images)
                
                # 获取预测类别
                _, preds = torch.max(outputs, 1)
                
                # 统计整体准确率
                batch_size = labels.size(0)
                total_test_samples += batch_size
                test_acc += get_accuracy(outputs, labels) * batch_size
                
                # 收集预测和真实标签
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # 统计每个类别的正确率
                corrects = (preds == labels).cpu().numpy()
                for j in range(batch_size):
                    label = labels[j].item()
                    class_correct[label] += corrects[j]
                    class_count[label] += 1
        
        # 计算整体测试准确率
        test_acc /= total_test_samples
        end_time = time.time()
        print(f"Testing time: {end_time - start_time:.2f}s")
        
        # 计算模型在测试集上的总参数（包含不可训练参数）
        test_params = sum(p.numel() for p in model.parameters())
        test_params_mb = test_params * 4 / (1024 ** 2)
        print(f"Total Parameters: {test_params_mb:.2f} MB")
        
        # 计算Cohen's Kappa系数（评估模型一致性的指标）
        kappa_score = cohen_kappa_score(all_targets, all_preds)
        print(f"Cohen's Kappa Score: {kappa_score:.4f}")
        
        # 计算并输出各类别准确率
        class_accuracy = class_correct / (class_count + 1e-9)  # 避免除以零
        print(f"Class-wise Accuracy: Mean={np.mean(class_accuracy)*100:.2f}%")
        
        # 打印epoch总结
        print(f'Epoch: {epoch} | Loss: {train_running_loss:.4f} | ' +
              f'Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | ' +
              f'Weights: {avg_fusion_weights}')
        
        # 更新学习率
        scheduler.step()
        
        # 保存模型检查点（每10个epoch）
        if (epoch + 1) % 10 == 0:
            save_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # 程序入口
    main()
