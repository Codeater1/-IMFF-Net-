# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化算法模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
from torchvision import models, transforms, datasets  # 计算机视觉模型和工具
from PIL import Image  # 图像处理库
import os  # 操作系统接口
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
import torch.nn.functional as F  # PyTorch函数式接口
import numpy as np  # 数值计算库
import time  # 时间模块
from sklearn.metrics import cohen_kappa_score  # Cohen's Kappa评估指标

# ===== 全局参数配置 =====
image_size = 224  # 输入图像尺寸
batch_size = 16  # 训练批量大小
num_classes = 21  # 分类任务类别数
num_epochs = 25  # 训练总轮数
learning_rate = 0.0002  # 初始学习率
min_lr = 1e-6  # 最小学习率（用于学习率调度）

# === LoG滤波器生成函数 ===
def get_log_kernel(size, sigma):
    """创建2D LoG(Laplacian of Gaussian)滤波器"""
    # 创建坐标系网格
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    y = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(x, y)  # 生成网格点
    
    # 计算LoG函数值
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (1 * sigma ** 2))
    kernel = kernel * (xx ** 2 + yy ** 2 - 1 * sigma ** 2) / (sigma ** 4)
    
    # 转换为PyTorch张量
    return torch.from_numpy(kernel.astype(np.float32))

# === LoG卷积层实现 ===
class LogLayer(nn.Module):
    """实现LoG(高斯拉普拉斯)卷积层"""
    def __init__(self, channels, size, sigma):
        super(LogLayer, self).__init__()
        # 创建并注册为不可训练参数
        self.weight = nn.Parameter(
            get_log_kernel(size, sigma).unsqueeze(0).unsqueeze(0), 
            requires_grad=False
        )
        self.groups = channels  # 分组卷积数（等于输入通道数）
    
    def forward(self, x):
        """前向传播"""
        # 将滤波器复制到每个输入通道
        weight = self.weight.repeat(self.groups, 1, 1, 1)
        # 应用分组卷积
        return F.conv2d(x, weight, padding=1, groups=self.groups)

# === 多尺度特征提取模块 ===
class MultiScaleBranchModule(nn.Module):
    """并行使用不同尺寸卷积核提取特征"""
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
        """前向传播"""
        branch2_out = self.branch2(x)  # 3x3卷积输出
        branch3_out = self.branch3(x)  # 5x5卷积输出
        # 沿通道维度拼接特征
        output = torch.cat([branch2_out, branch3_out], dim=1)
        return output

# === 对象特征提取分支 ===
class ObjectFeatureBranch(nn.Module):
    """提取对象级特征的分支网络"""
    def __init__(self, in_channels, out_channels, num_classes):
        super(ObjectFeatureBranch, self).__init__()
        # 多尺度特征提取
        self.multi_scale_branch = MultiScaleBranchModule(in_channels, out_channels)
        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        """前向传播"""
        x = self.multi_scale_branch(x)  # 多尺度特征
        x = self.pool(x)  # 全局池化
        return x

# === 加权特征融合层 ===
class WeightedFusionLayer(nn.Module):
    """两种特征的加权融合层"""
    def __init__(self):
        super(WeightedFusionLayer, self).__init__()
        # 可学习的权重参数
        self.weight_out4 = nn.Parameter(torch.tensor(0.5))  # 原始特征权重
        self.weight_out_log = nn.Parameter(torch.tensor(0.5))  # LoG特征权重
    
    def forward(self, out4, out_log):
        """前向传播"""
        # 简单的加权求和
        return self.weight_out4 * out4 + self.weight_out_log * out_log

# === 基于熵的关键区域选择模块 ===
class Entropy(nn.Module):
    """计算局部熵的特征图"""
    def __init__(self, kernel_size=2):
        super(Entropy, self).__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x):
        """前向传播计算局部熵"""
        b, c, h, w = x.size()
        n = self.kernel_size ** 2  # 局部区域像素数
        
        # 使用滑动窗口遍历特征图
        x = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        x = x.contiguous().view(b, c, -1, n)  # 重塑形状
        
        # 计算局部熵
        p = x / (torch.sum(x, dim=-1, keepdim=True) + 1e-6)  # 概率分布
        log_p = torch.log(p + 1e-6)  # 对数概率
        ent = -torch.sum(p * log_p, dim=-1)  # 熵值计算
        return ent

class EKLM(nn.Module):
    """基于熵的关键区域定位模块"""
    def __init__(self, kernel_size=3, ratio=0.5):
        super(EKLM, self).__init__()
        self.kernel_size = kernel_size
        self.ratio = ratio  # 保留比例
        self.entropy = Entropy(kernel_size)  # 熵计算模块
    
    def forward(self, x):
        """前向传播"""
        b, c, h, w = x.size()
        mask = torch.zeros_like(x)  # 创建掩码
        
        # 遍历批次中的每个样本
        for i in range(b):
            # 计算熵图
            ent = self.entropy(x[i].unsqueeze(0))
            
            # 选择熵值最高的区域
            k = int(self.ratio * ent.numel())  # 保留元素数量
            _, idx = torch.topk(ent.view(-1), k)  # 获取topk索引
            
            # 创建二值掩码
            mask_temp = torch.zeros_like(ent.view(-1)).scatter_(0, idx, 1).view_as(ent)
            
            # 还原到原始尺寸
            mask_temp = mask_temp.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            mask_temp = torch.max(mask_temp, dim=-1)[0].max(dim=-1)[0]
            mask_temp = F.interpolate(mask_temp.unsqueeze(1), x.size()[2:], mode='nearest')
            
            # 应用掩码
            mask[i] = mask_temp
        
        return x * mask  # 保留关键区域

# === 深度可分离卷积模块 ===
class DDP(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, num_classes):
        super(DDP, self).__init__()
        # 标准卷积
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # 深度可分离卷积（深度卷积+点卷积）
        self.conv2_depthwise = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv2_pointwise = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # 池化与分类
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # 输出分类
    
    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 深度可分离卷积
        x = self.conv2_depthwise(x)  # 深度卷积
        x = self.conv2_pointwise(x)  # 点卷积
        x = self.bn2(x)
        x = self.relu2(x)
        # 全局平均池化
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        return x

# === OmegaVisionNet主模型 ===
class OmegaVisionNet(nn.Module):
    """遥感图像分类模型"""
    def __init__(self, num_classes=21):
        super(OmegaVisionNet, self).__init__()
        # 加载预训练DenseNet121
        densenet121 = models.densenet121(pretrained=True)
        
        # === 自定义初始卷积层 ===
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, dilation=3)  # 空洞卷积
        self.bn1 = nn.BatchNorm2d(256)
        
        # === 复用DenseNet底层特征 ===
        self.features_conv0 = densenet121.features.conv0
        self.features_norm0 = densenet121.features.norm0
        self.features_relu0 = densenet121.features.relu0
        self.features_pool0 = densenet121.features.pool0
        
        # === 对象特征分支 ===
        self.extra_feature_branch = ObjectFeatureBranch(64, 128, num_classes)  # 浅层特征分支
        self.extra_feature_branch1 = ObjectFeatureBranch(256, 128, num_classes)  # 中层特征分支
        
        # === DenseNet主路径 ===
        self.features_denseblock1 = densenet121.features.denseblock1
        self.features_transition1 = densenet121.features.transition1
        self.features_denseblock2 = densenet121.features.denseblock2
        self.features_transition2 = densenet121.features.transition2
        self.features_denseblock3 = densenet121.features.denseblock3
        self.features_transition3 = densenet121.features.transition3
        self.features_denseblock4 = densenet121.features.denseblock4
        
        # === 特征增强与选择 ===
        self.entropy_crop = EKLM()  # 关键区域选择模块
        self.weighted_fusion = WeightedFusionLayer()  # 特征融合层
        self.features_norm5 = densenet121.features.norm5  # 归一化层
        
        # === 分类器 ===
        self.classifier = nn.Linear(1792, num_classes)  # 主分类器
        
        # === 辅助路径处理 ===
        self.ddp = DDP(num_classes)  # 深度可分离卷积模块
        self.log_layer = LogLayer(channels=1024, size=3, sigma=0.5)  # LoG特征提取
        self.fc = nn.Linear(64, num_classes)  # 辅助分类器
    
    def forward(self, x):
        """前向传播流程"""
        # 路径1: 自定义初始卷积路径
        go = self.conv1(x)
        go = self.bn1(go)
        go = self.features_relu0(go)
        go = self.features_pool0(go)
        
        # 路径2: 标准DenseNet前处理
        out0 = self.features_conv0(x)
        out0 = self.features_norm0(out0)
        out0 = self.features_relu0(out0)
        out0 = self.features_pool0(out0)
        
        # 对象特征提取
        out_branch = self.extra_feature_branch(out0)
        
        # DenseNet主路径处理
        out = self.features_denseblock1(out0)
        out = self.features_transition1(out)
        out = self.features_denseblock2(out)
        out = self.features_transition2(out)
        out_branch1 = self.extra_feature_branch1(out)
        out = self.features_denseblock3(out)
        out = self.features_transition3(out)
        out4 = self.features_denseblock4(out)  # 深层特征
        
        # 特征归一化与池化
        out5 = self.features_norm5(out4)
        out_pool = F.adaptive_avg_pool2d(out5, (1, 1))
        out_pool = out_pool.view(out_pool.size(0), -1)  # 展平
        
        # 路径1处理
        go_pool = F.adaptive_avg_pool2d(go, (1, 1))
        go_pool = go_pool.view(go_pool.size(0), -1)
        
        # LoG特征处理
        out_log = self.log_layer(out5)  # 应用LoG滤波器
        out_log = self.weighted_fusion(out5, out_log)  # 特征融合
        
        # 关键区域选择
        oute = self.entropy_crop(out4)  # 基于熵选择区域
        
        # 辅助路径处理
        out_custom = self.ddp(out_log)  # 处理原始+LoG特征
        out_custom1 = self.ddp(oute)  # 处理关键区域特征
        
        # 辅助分类
        out_custom = self.fc(out_custom)
        out_custom1 = self.fc(out_custom1)
        
        # 特征融合与主分类
        combined = torch.cat([
            out_pool,
            out_branch1.view(out_branch1.size(0), -1), 
            out_branch.view(out_branch.size(0), -1),
            go_pool
        ], dim=1)
        out_class = self.classifier(combined)  # 主分类输出
        
        # 三路输出加权融合
        out = 0.5 * out_class + 0.3 * out_custom + 0.2 * out_custom1
        return out

# === 数据增强: 随机亮度和对比度调整 ===
class RandomBrightnessContrast(transforms.RandomApply):
    """自定义亮度对比度增强"""
    def __init__(self, brightness=0.1, contrast=0.2, p=0.5):
        transform = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        super().__init__(transforms=[transform], p=p)

# === 训练数据预处理 ===
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整尺寸
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    RandomBrightnessContrast(),  # 随机亮度和对比度调整
    transforms.RandomCrop(image_size, padding=1),  # 随机裁剪
    transforms.ToTensor(),  # 转为张量
    # ImageNet标准化参数
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 测试数据预处理 ===
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整尺寸
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 遥感图像数据集类 ===
class AID20Dataset(Dataset):
    """遥感图像数据集加载器"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))  # 获取类别列表
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 类别到索引的映射
        self.filepaths = []  # 存储图像路径
        self.labels = []  # 存储对应标签
        
        # 构建数据集索引
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for file in os.listdir(cls_dir):
                self.filepaths.append(os.path.join(cls_dir, file))
                self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        """数据集大小"""
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert("RGB")  # 确保RGB格式
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)  # 应用变换
        
        return img, label

# === 设置数据集路径 ===
data_dir_train = '/root/autodl-tmp/Swin-Transformer-main/ucm_train'  # 训练集路径
data_dir_test = '/root/autodl-tmp/Swin-Transformer-main/ucm_test'  # 测试集路径

# === 创建数据集和数据加载器 ===
train_dataset = AID20Dataset(data_dir=data_dir_train, transform=train_transform)
test_dataset = AID20Dataset(data_dir=data_dir_test, transform=test_transform)

train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    num_workers=4, 
    shuffle=True  # 训练时打乱数据
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    num_workers=4,  
    shuffle=False  # 测试时不打乱
)

# === 初始化模型 ===
# 模型1: 自定义模型
model = OmegaVisionNet(num_classes=num_classes)

# 模型2: DenseNet121预训练模型
densenet_model = models.densenet121(pretrained=True)
densenet_model.classifier = nn.Linear(
    densenet_model.classifier.in_features, 
    num_classes
)

# === 训练配置 ===
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=min_lr)  # 学习率调度

# === GPU支持 ===
if torch.cuda.is_available():
    model = model.cuda()
    densenet_model = densenet_model.cuda()

def get_accuracy(logit, target):
    """计算分类准确率"""
    # 获取预测类别（最大概率的索引）
    preds = torch.max(logit, 1)[1].view(target.size())
    # 计算正确数量
    corrects = (preds == target.data).float().sum()
    # 计算准确率百分比
    accuracy = 100.0 * corrects / target.size(0)
    return accuracy.item()

# === 类别准确率追踪 ===
class_count = np.zeros(num_classes)  # 每个类别的样本数
class_correct = np.zeros(num_classes)  # 每个类别的正确预测数

# === 训练主循环 ===
for epoch in range(num_epochs):
    model.train()  # 训练模式
    train_running_loss = 0.0  # 训练损失累计
    train_acc = 0.0  # 训练准确率累计
    total_train_samples = 0  # 训练样本计数
    
    # 遍历训练批次
    for i, (images, labels) in enumerate(train_loader):
        # GPU支持
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # 开始计时
        start_time = time.time()
        
        # 优化器梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播与参数更新
        loss.backward()
        optimizer.step()
        
        # 结束计时
        end_time = time.time()
        
        # 统计指标
        batch_size = labels.size(0)
        total_train_samples += batch_size
        train_running_loss += loss.detach().item() * batch_size
        train_acc += get_accuracy(outputs, labels) * batch_size
    
    # 计算epoch平均训练指标
    train_running_loss /= total_train_samples
    train_acc /= total_train_samples
    
    # === 评估模式 ===
    model.eval()
    test_acc = 0.0
    total_test_samples = 0
    all_preds = []  # 所有预测值
    all_targets = []  # 所有真实标签
    
    # 遍历测试批次
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # 开始计时
        start_time = time.time()
        
        # 前向传播（无梯度计算）
        outputs = model(images)
        
        # 结束计时
        end_time = time.time()
        
        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).squeeze()
        
        # 统计类别准确率
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += correct[i].item()
            class_count[label] += 1
        
        # 统计整体准确率
        batch_size = labels.size(0)
        total_test_samples += batch_size
        test_acc += get_accuracy(outputs, labels) * batch_size
        
        # 收集评估结果
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    
    # 计算测试准确率
    test_acc /= total_test_samples
    
    # 计算Cohen's Kappa系数
    kappa_score = cohen_kappa_score(all_targets, all_preds)
    print("Cohen's Kappa Score: %.4f" % kappa_score)
    
    # 高准确率时打印类别详细结果
    if test_acc > 94.6:
        # 打印每个类别准确率
        for i in range(num_classes):
            class_name = f"Class_{i}"
            accuracy = 100 * class_correct[i] / (class_count[i] + 1e-5)
            print('Accuracy of class %s : %2d %%' % (class_name, accuracy))
    
    # 高准确率时保存模型
    if test_acc > 94:
        torch.save(model.state_dict(), '/root/autodl-tmp/UCM/model_new.pth')
        print("Model saved!")
    
    # 打印epoch总结
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f | Test Accuracy: %.2f' \
          %(epoch, train_running_loss, train_acc, test_acc))
    
    # 更新学习率
    scheduler.step()
