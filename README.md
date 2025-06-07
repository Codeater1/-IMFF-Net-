# -IMFF-Net-
Using DenseNet121 as the backbone, the model is optimized to achieve high classification accuracy with a low parameter count.

# Description
IMFF_NET_Unlearnable.py and IMFF_Net_learnable.py contain all model code (including various modules).

IMFF_NET_Unlearnable.py: Model with ​​fixed-ratio​​, ​​non-trainable​​ output fusion layer.
IMFF_Net_learnable.py: Model with ​​trainable weights​​ added to the output fusion layer.
data.py: Download links for datasets used.
model_pretrained.py: Pre-trained model weights.
​​LOGM Module​​ (coded as LogLayer).
​​HECM Module​​ (coded as EKLM).
​​IMFF-NET​​ (coded as OmegaVisionNet).
​​DIM​​ (coded as DDP).
# Overall Model Framework
![b6c472d5-b302-4679-a8d1-d23da8882843](https://github.com/user-attachments/assets/5f89894a-9677-4a49-8b55-9edd20649c18)


