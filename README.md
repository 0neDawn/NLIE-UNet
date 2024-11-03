# NLIE-UNet
Dynamic Neighborhood-Enhanced UNet with Interwoven Fusion for Medical lmage Segmentation.
## Introduction
While Convolutional Neural Networks (CNNs), particularly UNet and its variants, have demonstrated excellent performance, they face challenges such as insufficient adaptive capability, redundant feature information, and weak multi-scale feature sensing. To address these issues, we propose a novel medical image segmentation method called Dynamic Neighborhood-Enhanced UNet with Interwoven Fusion (NLIE-UNet).Our approach includes a Cyclic Dynamic Convolution Block (CDCB) to adaptively capture edge contour information, a Neighborhood Enhanced Bridge (NEB) to exploit the consistency and complementarity of different layer features, and a Hierarchical Interwoven Fusion Module (HIFM) to fuse cross-layer information effectively. Extensive experiments on five public medical image datasets demonstrate the effectiveness and superiority of our method.
<img width="697" alt="image" src="https://github.com/user-attachments/assets/22adc429-30c8-4e7e-a745-bc94cc0f2abd"> 
## Environment
Our code framework is based on UNeXt；

To install all the dependencies using conda:

'''conda env create -f environment.yml

conda activate NLIE_UNet'''

If you prefer pip, install following versions:

'''timm==0.3.2

mmcv-full==1.2.7

torch==1.7.1

torchvision==0.8.2

opencv-python==4.5.1.48'''

## Datasets

'''BUSI: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/

Kvasir-SEG: https://datasets.simula.no/kvasir-seg/

GlaS: https://github.com/McGregorWwww/UCTransNet/

2018 DSB: https://www.kaggle.com/c/data-science-bowl-2018/'''

