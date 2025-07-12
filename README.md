# Rep-Mamba: Re-parameterization in Vision Mamba  for Lightweight Remote Sensing Image  Super-Resolution
## Abstract
> Selective space model (Mamba) has recently demonstrated great potential in remote sensing image super-resolution (RSISR) tasks due to its capability for long-range dependency modeling with linear computational complexity. Despite these merits, existing Mamba architectures face two critical challenges in large-scale remote sensing scenarios: i) Neglecting the local semantic integrity due to the unfolding 1D sequential representations; ii) Facing the dilemma between effectiveness and efficiency. To address these issues, we propose Rep-Mamba, a lightweight progressive multi-scale feature fusion architecture based on the State Space Model (SSM) for RSISR. Specifically, we innovatively design a Cross-Scale State Propagation (CSSP) mechanism and construct a Lightweight Progressive Fusion Module (LPFM) to dynamically capture hierarchical spatial dependencies in remote sensing scenes while maintaining high computational efficiency. Moreover, to achieve synergistic optimization between local semantic structure preservation and global context modeling, we introduce differentiable Re-parameterization Convolution (RepConv), which significantly enhances reconstruction accuracy and visual quality without compromising computational efficiency. Extensive experiments across multiple benchmarks demonstrate that Rep-Mamba achieves a superior trade-off between accuracy and complexity, highlighting its effectiveness and scalability.
## Network  
 ![image](/fig/network.png)
 
## 🧩 Install
```
git clone https://github.com/meigeni0929/Rep-Mamba.git
```

## Environment
 > * CUDA 11.8
 > * Python 3.8
 > * PyTorch 2.0.0
 > * Torchvision 0.15.1
 > * basicsr 1.4.2 

## 🎁 Dataset
Please download the following remote sensing benchmarks:
| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) | [DIOR](https://www.sciencedirect.com/science/article/pii/S0924271619302825) | [NWPU-RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544)
| :----: | :-----: | :----: | :----: | :----: |
|Training | [Download](https://captain-whu.github.io/AID/) | None | None | None |
|Testing | [Download](https://captain-whu.github.io/AID/) | [Download](https://captain-whu.github.io/DOTA/dataset.html) | [Download](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) | [Download](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp)

## 🧩 Test
- **Step I.**  Use the structure below to prepare your dataset.

/xxxx/xxx/ (your data path)
```
/GT/ 
   /000.png  
   /···.png  
   /099.png  
/LR/ 
   /000.png  
   /···.png  
   /099.png  
```
- **Step II.**  Change the `--data_dir` to your data path.
- **Step III.**  Run the eval_4x.py
```
python eval_4x.py
```

### Train
```
python train.py
```




## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: 24s103263@stu.hit.edu.cn

