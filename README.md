# -IMFF-Net-
以DenseNet121为Backbone，对模型进行优化，以实现在低参数量情况下获得高分类精度的模型
# 说明
IMFF_NET_Unlearnable.py以及IMFF_Net_learnable.py保存了模型的全部代码（包括模型的各种模块）
1.IMFF_NET_Unlearnable.py是输出融合层固定比例的、不可学习的模型
2.IMFF_Net_learnable.py是输出融合层导入了可训练权重模块的模型
3.data.py记录所使用的数据集的下载地址
4.model_pretrained.py记录训练好的模型的权重

# LOGM模块（对应代码中的LogLayer）

# HECM模块（对应代码中的EKLM）

# IMFF-NET（对应代码中的OmegaVisionNet）

# DIM(对应代码中的DDP)
