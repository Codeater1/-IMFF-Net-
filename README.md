# -IMFF-Net-
Using DenseNet121 as the backbone, the model is optimized to achieve high classification accuracy with a low parameter count.

# Description
IMFF_NET_Unlearnable.py and IMFF_Net_learnable.py contain all model code (including various modules).  
IMFF_NET_Unlearnable.py: logm is a non-learnable module.  
IMFF_Net_learnable.py: logm is a learnable module.  
data.py: Download links for datasets used.  
model_pretrained.py: Pre-trained model weights.  
​​1.LOGM Module​​ (coded as LogLayer).  
​​2.HECM Module​​ (coded as EKLM).  
​​3.IMFF-NET​​ (coded as OmegaVisionNet).  
​​4.DIM​​ (coded as DDP).



