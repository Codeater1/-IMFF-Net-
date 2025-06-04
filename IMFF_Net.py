# 导入必要的库
import torch
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # 优化算法模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
from torchvision import models, transforms, datasets  # 计算机视觉相关模块
from torch.utils.data import Subset  # 数据集子集
from PIL import Image  # 图像处理库
import os  # 操作系统接口
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
import torch.nn.functional as F  # PyTorch函数接口
from sklearn.metrics import confusion_matrix  # 混淆矩阵
import matplotlib.pyplot as plt  # 绘图库
import seaborn as sns  # 数据可视化库
import numpy as np  # 数值计算库
from scipy import ndimage  # 图像处理库
import pandas as pd  # 数据处理库
import time  # 时间模块
from sklearn.metrics import cohen_kappa_score  # Cohen's Kappa评分
from sklearn.manifold import TSNE  # t-SNE降维算法

# ========================== 超参数设置 ==========================
image_size = 224  # 输入图像尺寸
batch_size = 16  # 批处理大小
num_classes = 30  # 分类类别数
num_epochs = 30  # 训练轮数
learning_rate = 0.0002  # 初始学习率
min_lr = 1e-6  # 最小学习率

# ========================== 模型组件定义 ==========================

def get_log_kernel(size, sigma):
    """
    生成2D LoG(Laplacian of Gaussian)滤波器核
    参数:
        size: 滤波器核尺寸
        sigma: 高斯核的标准差
    返回:
        二维LoG滤波器核(tensor)
    """
    # 创建坐标网格
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    y = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(x, y)
    
    # 计算LoG核
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (1 * sigma ** 2))  # 高斯分量
    kernel = kernel * (xx ** 2 + yy ** 2 - 1 * sigma ** 2) / (sigma ** 4)  # 拉普拉斯分量
    
    return torch.from_numpy(kernel.astype(np.float32))

class LogLayer(nn.Module):
    """
    LoG(Laplacian of Gaussian)特征提取层
    通过LoG滤波器提取特征
    """
    def __init__(self, channels, size, sigma):
        """
        初始化LoG层
        参数:
            channels: 输入通道数
            size: 滤波器尺寸
            sigma: 高斯标准差
        """
        super(LogLayer, self).__init__()
        # 创建不可训练的LoG滤波器权重
        self.weight = nn.Parameter(get_log_kernel(size, sigma).unsqueeze(0).unsqueeze(0), 
                                  requires_grad=False)
        self.groups = channels  # 组卷积的分组数（与通道数相同）
        
    def forward(self, x):
        # 将滤波器复制到与输入通道数匹配
        weight = self.weight.repeat(self.groups, 1, 1, 1)
        # 应用卷积操作（组卷积形式）
        return F.conv2d(x, weight, padding=1, groups=self.groups)

class MultiScaleBranchModule(nn.Module):
    """
    多尺度特征提取分支
    并行使用不同卷积核大小提取特征
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBranchModule, self).__init__()
        # 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        # 沿通道维度拼接特征
        output = torch.cat([branch2_out, branch3_out], dim=1)
        return output

class ObjectFeatureBranch(nn.Module):
    """
    对象特征提取分支
    使用多尺度特征提取+全局池化
    """
    def __init__(self, in_channels, out_channels, num_classes):
        super(ObjectFeatureBranch, self).__init__()
        # 多尺度特征提取
        self.multi_scale_branch = MultiScaleBranchModule(in_channels, out_channels)
        # 自适应平均池化
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.multi_scale_branch(x)  # 提取多尺度特征
        x = self.pool(x)  # 全局池化
        return x
    
class WeightedFusionLayer(nn.Module):
    """
    加权特征融合层
    学习两组特征图的最佳融合权重
    """
    def __init__(self):
        super(WeightedFusionLayer, self).__init__()
        # 初始化两个可学习的权重参数
        self.weight_out4 = nn.Parameter(torch.tensor(0.5))
        self.weight_out_log = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, out4, out_log):
        # 使用sigmoid确保权重在0到1之间
        w_out4 = torch.sigmoid(self.weight_out4)
        w_out_log = torch.sigmoid(self.weight_out_log)
        
        # 归一化权重，使两者之和为1
        total_weight = w_out4 + w_out_log
        w_out4 = w_out4 / total_weight
        w_out_log = w_out_log / total_weight
        
        # 返回加权融合的特征图
        return w_out4 * out4 + w_out_log * out_log
    
class LearnableWeightsFusion(nn.Module):
    """
    可学习权重输出融合层
    融合多个分类器的预测结果
    """
    def __init__(self, num_outputs=3, random_init=False):
        """
        初始化融合层
        参数:
            num_outputs: 需要融合的分类器数量
            random_init: 是否随机初始化权重
        """
        super(LearnableWeightsFusion, self).__init__()
        
        if random_init:
            # 随机初始化logits值
            std = 1.0 / np.sqrt(num_outputs)
            initial_weights = torch.randn(num_outputs) * std
        else:
            # 使用期望权重[0.5, 0.2, 0.3]进行初始化
            desired_weights = torch.tensor([0.5, 0.2, 0.3])
            logits = torch.zeros(num_outputs, requires_grad=True)
            # 使用LBFGS优化器找到合适的初始值
            optimizer = optim.LBFGS([logits], lr=1)
            for _ in range(100):
                def closure():
                    optimizer.zero_grad()
                    current_weights = F.softmax(logits, dim=0)
                    loss = ((current_weights - desired_weights) ** 2).sum()
                    loss.backward()
                    return loss
                optimizer.step(closure)
            initial_weights = logits.detach().clone()
        
        # 注册为可学习参数
        self.logits = nn.Parameter(initial_weights)
        
    def forward(self, out_class, out_custom, out_custom1):
        """
        前向传播
        返回:
            out: 加权融合的输出
            weights: 当前融合权重
        """
        # 计算softmax权重
        weights = F.softmax(self.logits, dim=0)
        # 加权融合三个分类器输出
        out = weights[0] * out_class + weights[1] * out_custom + weights[2] * out_custom1
        return out, weights

# ========================== 特征分析算法 ==========================

def compute_entropy(feature_map, xs, xd, ys, yd):
    """
    计算特征图指定区域的熵
    参数:
        feature_map: 输入特征图
        xs, xd: x方向起始和结束索引
        ys, yd: y方向起始和结束索引
    返回:
        区域熵值
    """
    # 裁剪特征图区域
    cropped_feature_map = feature_map[:, :, xs:xd, ys:yd]
    # 展平特征
    flattened_feature_map = cropped_feature_map.view(-1)
    # 计算直方图
    histogram = torch.histc(flattened_feature_map, bins=256, min=0, max=1)
    # 计算概率分布
    prob_distribution = histogram / torch.sum(histogram)
    # 计算信息熵
    entropy = -torch.sum(prob_distribution * torch.log2(prob_distribution + 1e-9))
    return entropy

def compute_entropy_for_each_y(feature_map, xs, xd):
    """
    计算特征图在x方向切片上沿y方向的熵
    参数:
        feature_map: 输入特征图
        xs, xd: x方向切片范围
    返回:
        包含各y位置熵的张量
    """
    entropies = []
    # 遍历每个y位置
    for ys in range(feature_map.shape[3]):
        entropy = compute_entropy(feature_map, xs, xd, ys, ys + 1)
        entropies.append(entropy)
    return torch.tensor(entropies)

def EKLM(A_hat, xs, T):
    """
    EKLM(Entropy-based Key Localization Method)算法
    基于信息熵的关键区域定位方法
    
    参数:
        A_hat: 输入特征图
        xs: 起始x坐标
        T: 熵阈值比例
        
    返回:
        [xs, xd, ys, yd] 关键区域边界坐标
    """
    xs = 0  # 固定从x=0开始
    # 计算y方向熵分布
    entropies = compute_entropy_for_each_y(A_hat, xs, xs + 1)
    # 找到熵最大的y位置
    ys = torch.argmax(entropies)
    yd = ys + 1
    xd = xs + 1
    
    # 计算当前区域的熵比例
    total_entropy = compute_entropy(A_hat, 0, A_hat.shape[2], 0, A_hat.shape[3])
    Ts = compute_entropy(A_hat, xs, xd, ys, yd) / total_entropy
    
    # 扩展区域直到达到熵阈值
    while Ts < T:
        # 尝试向右扩展
        if (xd + 1 < A_hat.shape[2] and 
            compute_entropy(A_hat, xs, xd + 1, ys, yd) > compute_entropy(A_hat, xs, xd, ys, yd)):
            xd += 1
        # 尝试向上扩展
        elif (ys - 1 >= 0 and 
              compute_entropy(A_hat, xs, xd, ys - 1, yd) > compute_entropy(A_hat, xs, xd, ys, yd)):
            ys -= 1
        # 尝试向下扩展
        elif (yd + 1 < A_hat.shape[3] and 
              compute_entropy(A_hat, xs, xd, ys, yd + 1) > compute_entropy(A_hat, xs, xd, ys, yd)):
            yd += 1
        else:  
            break  # 无法进一步扩展
        
        # 更新当前熵比例
        Ts = compute_entropy(A_hat, xs, xd, ys, yd) / total_entropy
        
    return [xs, xd, ys, yd]

# ========================== 网络模块 ==========================

class DDP(nn.Module):
    """
    DDP(Depthwise Decoupled Processor)模块
    深度可分离卷积处理器
    """
    def __init__(self, num_classes):
        super(DDP, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # 深度可分离卷积
        self.conv2_depthwise = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv2_pointwise = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

class OmegaVisionNet(nn.Module):
    """
    OmegaVisionNet主网络结构
    基于DenseNet121的增强遥感图像分类网络
    """
    def __init__(self, num_classes=30):
        super(OmegaVisionNet, self).__init__()
        # 加载预训练的DenseNet121
        densenet121 = models.densenet121(pretrained=True)
        
        # 自定义输入处理层
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, dilation=3)
        self.bn1 = nn.BatchNorm2d(256)
        
        # 重用DenseNet的低级特征提取层
        self.features_conv0 = densenet121.features.conv0
        self.features_norm0 = densenet121.features.norm0
        self.features_relu0 = densenet121.features.relu0
        self.features_pool0 = densenet121.features.pool0
        
        # 多尺度特征提取分支
        self.extra_feature_branch = ObjectFeatureBranch(64, 128, num_classes)
        self.extra_feature_branch1 = ObjectFeatureBranch(256, 128, num_classes)
        
        # 重用DenseNet的高层模块
        self.features_denseblock1 = densenet121.features.denseblock1
        self.features_transition1 = densenet121.features.transition1
        self.features_denseblock2 = densenet121.features.denseblock2
        self.features_transition2 = densenet121.features.transition2
        self.features_denseblock3 = densenet121.features.denseblock3
        self.features_transition3 = densenet121.features.transition3
        self.features_denseblock4 = densenet121.features.denseblock4
        
        # 特征融合层
        self.weighted_fusion = WeightedFusionLayer()
        self.features_norm5 = densenet121.features.norm5
        
        # 分类器
        self.classifier = nn.Linear(1792, num_classes)
        
        # 自定义处理模块
        self.ddp = DDP(num_classes)
        self.log_layer = LogLayer(channels=1024, size=3, sigma=0.5)
        self.fc = nn.Linear(64, num_classes)
        
        # 输出融合层
        self.output_fusion = LearnableWeightsFusion(3)
        
    def forward(self, x):
        """前向传播过程"""
        # 路径1: 自定义卷积路径
        go = self.conv1(x)
        go = self.bn1(go)
        go = self.features_relu0(go)
        go = self.features_pool0(go)
        
        # 路径2: DenseNet标准路径
        out0 = self.features_conv0(x)
        out0 = self.features_norm0(out0)
        out0 = self.features_relu0(out0)
        out0 = self.features_pool0(out0)
        
        # 早期特征提取分支
        out_branch = self.extra_feature_branch(out0)
        
        # DenseNet主路径
        out = self.features_denseblock1(out0)
        out = self.features_transition1(out)
        out = self.features_denseblock2(out)
        out = self.features_transition2(out)
        
        # 中层特征提取分支
        out_branch1 = self.extra_feature_branch1(out)
        
        out = self.features_denseblock3(out)
        out = self.features_transition3(out)
        out4 = self.features_denseblock4(out)  # 获取高层特征用于区域定位
        
        # 主要分类路径
        out5 = self.features_norm5(out4)
        out_pool = F.adaptive_avg_pool2d(out5, (1, 1))
        out_pool = out_pool.view(out_pool.size(0), -1)
        
        # 处理自定义路径
        go_pool = F.adaptive_avg_pool2d(go, (1, 1))
        go_pool = go_pool.view(go_pool.size(0), -1)
        
        # LoG特征处理路径
        out_log = self.log_layer(out5)
        out_log = self.weighted_fusion(out5, out_log)  # 融合原始特征和LoG特征
                             
        # 使用EKLM算法定位关键区域
        xs = 0
        T = 0.4  # 熵比例阈值
        xs, xd, ys, yd = EKLM(out4.detach(), xs, T)  # detach防止梯度传播
        oute = out4[:, :, xs:xd, ys:yd]  # 裁剪关键区域特征
        
        # 处理两个自定义路径的特征
        out_custom = self.ddp(out_log)
        out_custom1 = self.ddp(oute)
        
        # 分类器输出
        out_custom = self.fc(out_custom)
        out_custom1 = self.fc(out_custom1)
        
        # 连接所有特征用于主分类器
        fused_out = torch.cat([out_pool, 
                              out_branch1.view(out_branch1.size(0), -1), 
                              out_branch.view(out_branch.size(0), -1), 
                              go_pool], dim=1)
        out_class = self.classifier(fused_out)
        
        # 融合三个分类器的输出
        fused_output, fusion_weights = self.output_fusion(out_class, out_custom, out_custom1)
        
        return fused_output, fusion_weights

# ========================== 数据预处理 ==========================

class RandomBrightnessContrast(transforms.RandomApply):
    """随机亮度和对比度变换"""
    def __init__(self, brightness=0.1, contrast=0.2, p=0.5):
        super().__init__(transforms=[transforms.ColorJitter(
            brightness=brightness, contrast=contrast)], p=p)

# 训练数据增强
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整尺寸
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    RandomBrightnessContrast(),  # 随机亮度和对比度变化
    transforms.RandomCrop(image_size, padding=1),  # 随机裁剪
    transforms.ToTensor(),  # 转为Tensor
    # 标准化(RGB三通道)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试数据预处理
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class AID20Dataset(Dataset):
    """遥感图像数据集加载器"""
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        参数:
            data_dir: 数据目录路径
            transform: 数据预处理变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))  # 获取类别列表
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 收集所有图像路径和标签
        self.filepaths = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for file in os.listdir(cls_dir):
                self.filepaths.append(os.path.join(cls_dir, file))
                self.labels.append(self.class_to_idx[cls])
                
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert("RGB")  # 确保RGB格式
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ========================== 工具函数 ==========================

def get_accuracy(logit, target):
    """
    计算分类准确率
    参数:
        logit: 模型输出logits
        target: 真实标签
    返回:
        准确率百分比
    """
    # 获取预测类别
    preds = torch.max(logit, 1)[1]
    # 计算正确数量
    corrects = (preds.view(target.size()) == target).sum().float()
    # 计算准确率
    accuracy = 100.0 * corrects / target.size(0)
    return accuracy.item()

# ========================== 主训练函数 ==========================

def main():
    # 设置训练参数
    global batch_size, num_classes, num_epochs, learning_rate, min_lr
    
    # 数据集路径
    data_dir_train = '/root/autodl-tmp/Swin-Transformer-main/50'
    data_dir_test = '/root/autodl-tmp/Swin-Transformer-main/50_test'
    
    # 创建数据集和数据加载器
    train_dataset = AID20Dataset(data_dir=data_dir_train, transform=train_transform)
    test_dataset = AID20Dataset(data_dir=data_dir_test, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             num_workers=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            num_workers=4, shuffle=False, pin_memory=True)
    
    # 初始化模型
    model = OmegaVisionNet(num_classes=num_classes)
    
    # 加载预训练权重（使用非严格模式）
    model_weights_path = '/root/autodl-tmp/AID20_Model/modelaid20.pth'
    model.load_state_dict(torch.load(model_weights_path), strict=False)
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=min_lr)
    
    # 使用GPU加速
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 训练循环
    fusion_weights_history = []  # 记录权重变化

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        model.train()  # 训练模式
        
        # 初始化统计变量
        train_running_loss = 0.0
        train_acc = 0.0
        total_train_samples = 0
        epoch_fusion_weights = []
        
        # 训练批次循环
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs, fusion_weights = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            batch_size = labels.size(0)
            total_train_samples += batch_size
            train_running_loss += loss.item() * batch_size
            train_acc += get_accuracy(outputs, labels) * batch_size
            
            # 记录融合权重
            epoch_fusion_weights.append(fusion_weights.detach().cpu().numpy())
            
        # 计算本轮训练的平均值
        train_running_loss /= total_train_samples
        train_acc /= total_train_samples
        avg_fusion_weights = np.mean(epoch_fusion_weights, axis=0)
        fusion_weights_history.append(avg_fusion_weights)
        
        # 计算训练时间
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f}s")
        
        # 计算模型大小
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_params_mb = train_params * 4 / (1024 ** 2)  # 估算内存占用
        print(f"Trainable Parameters: {train_params_mb:.2f} MB")
        
        # 输出当前融合权重
        print(f"Current fusion weights: {avg_fusion_weights}")
        
        # ====================== 测试阶段 ======================
        start_time = time.time()
        model.eval()  # 评估模式
        
        # 初始化测试统计变量
        test_acc = 0.0
        total_test_samples = 0
        all_preds = []  
        all_targets = []
        class_count = np.zeros(num_classes)  
        class_correct = np.zeros(num_classes)
        
        # 禁用梯度计算
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播
                outputs, _ = model(images)
                
                # 获取预测结果
                _, predicted = torch.max(outputs, 1)
                
                # 更新统计信息
                batch_size = labels.size(0)
                total_test_samples += batch_size
                test_acc += get_accuracy(outputs, labels) * batch_size
                
                # 收集预测和标签用于后续评估
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # 计算各类别准确率
                correct_mask = (predicted == labels).cpu().numpy()
                for i, label in enumerate(labels.cpu().numpy()):
                    class_count[label] += 1
                    class_correct[label] += correct_mask[i]
        
        # 计算测试统计量
        test_acc /= total_test_samples
        end_time = time.time()
        print(f"Testing time: {end_time - start_time:.2f}s")
        
        # 计算模型在测试集上的内存占用
        test_params = sum(p.numel() for p in model.parameters())
        test_params_mb = test_params * 4 / (1024 ** 2)
        print(f"Total Parameters: {test_params_mb:.2f} MB")
        
        # 计算Cohen's Kappa评分
        kappa_score = cohen_kappa_score(all_targets, all_preds)
        print(f"Cohen's Kappa Score: {kappa_score:.4f}")
        
        # 计算并输出各类别准确率
        class_accuracy = class_correct / class_count
        print(f"Class-wise Accuracy: {np.mean(class_accuracy):.2f}%")
        
        # 打印epoch总结
        print(f'Epoch: {epoch} | Loss: {train_running_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | '
              f'Weights: {avg_fusion_weights}')

        # 更新学习率
        scheduler.step()
        
        # 保存模型检查点
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()