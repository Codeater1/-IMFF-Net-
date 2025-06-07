# -IMFF-Net-
以DenseNet121为Backbone，对模型进行优化，以实现在低参数量情况下获得高分类精度的模型
# 说明
IMFF_NET_Unlearnable.py以及IMFF_Net_learnable.py保存了模型的全部代码（包括模型的各种模块）
1.IMFF_NET_Unlearnable.py是输出融合层固定比例的、不可学习的模型
2.IMFF_Net_learnable.py是输出融合层导入了可训练权重模块的模型
3.data.py记录所使用的数据集的下载地址
4.model_pretrained.py记录训练好的模型的权重
5.LOGM模块（对应代码中的LogLayer）
6.HECM模块（对应代码中的EKLM）
7.IMFF-NET（对应代码中的OmegaVisionNet）
8.DIM(对应代码中的DDP)
# 模型总框架
![b6c472d5-b302-4679-a8d1-d23da8882843](https://github.com/user-attachments/assets/0a5ad0b0-5048-44da-a69b-b5f8ca5c028b)

