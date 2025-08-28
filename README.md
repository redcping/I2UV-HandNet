# I2UV-HandNet: Image-to-UV Prediction Network for Accurate and High-Fidelity 3D Hand Mesh Modeling (ICCV 2021)  
![license](https://img.shields.io/badge/License-MIT-brightgreen)  ![python](https://img.shields.io/badge/Python-3.7-blue)  ![pytorch](https://img.shields.io/badge/PyTorch-1.7-orange)  



## 🗂️ About this Repository
This repository contains the (partially preserved) official code of our ICCV 2021 [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_I2UV-HandNet_Image-to-UV_Prediction_Network_for_Accurate_and_High-Fidelity_3D_Hand_ICCV_2021_paper.pdf):  
**“I2UV-HandNet: Image-to-UV Prediction Network for Accurate and High-Fidelity 3D Hand Mesh Modeling.”**

<div align="center">
<img width="689" height="243" alt="image" src="https://github.com/user-attachments/assets/21c61eee-9c41-42a7-9354-e1bfb40f8779" />
</div>

> 📜 **A story from 2021**:  
> While cleaning up old storage devices in 2025, we surprisingly found a forgotten USB drive.  
> Inside it was the original training code of our ICCV 2021 paper. The code may be **incomplete and unpolished**, but since the project marked an early step toward high-fidelity 3D hand reconstruction, we decided to open-source it in the hope that it could still inspire and help the community.  

---

## 📄 Description
Reconstructing an accurate and high-fidelity **3D human hand mesh from a single RGB image** is challenging due to diverse hand poses and severe occlusions.  

I2UV-HandNet introduced the **first UV-based 3D hand representation for hands**, allowing efficient and high-quality mesh recovery:  

- **UV-based Hand Representation**  
  - Represented 3D hand surfaces in **UV position maps** instead of MANO parameters or vertex regressions.  
  - Preserved dense spatial details for mesh recovery.  

- **AffineNet**  
  - An image-to-UV translation network with a novel **affine connection module** to resolve coordinate ambiguity between RGB and UV domains.  
  - Predicts a coarse UV position map from a single RGB input.  

- **SRNet (Super-Resolution for Hands)**  
  - Upsamples coarse UV maps to **high-resolution hand meshes**.  
  - Trained on a high-quality scan dataset (**SuperHandScan**), enabling detailed surface reconstruction.


---

## 📊 Main Results

- On **HO3D v2 dataset**, our method (submitted under the name **algcd**) ranked **1st place on the official HO3D Challenge leaderboard at that time**, demonstrating strong robustness under hand-object interactions.  

🔗 [HO3D V2 Challenge Leaderboard](https://competitions.codalab.org/competitions/22485#results)  

<div align="center">
 <img width="1134" height="729" alt="image" src="https://github.com/user-attachments/assets/7587607b-f1de-4fec-ba39-f65be9551beb" />
</div>


---

## ⚙️ Requirements
- Python ≥ 3.7  
- PyTorch ≥ 1.7  
- torchvision, numpy, tqdm  
- (Optional) [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/), [HO3D](https://competitions.codalab.org/competitions/22485) datasets for training & evaluation  

---

## 🚀 Usage

### Training Pipeline
The full training is a **two-step process**:  

1. **Train AffineNet (hand pose & shape estimation)**  
   ```
   sh run_hand.sh
   ```

2. **Train SRNet (hand mesh super-resolution, with AffineNet frozen))**  
   ```
   sh run_super.sh
   ```
   In this stage, the pretrained **AffineNet** is kept **frozen** to preserve its coarse UV prediction capability.  **SRNet** is trained on top of these fixed features to upsample UV maps and recover high-fidelity details.  

⚠️ **Note**: Since this code was recovered from an old archive, some parts may be **incomplete or require manual fixes** (e.g., dataset paths, model configs). Please adapt as needed for your environment.  

---

## 🎨 UV Design

The base UV mapping follows MANO’s right-hand topology (`handRightUV.obj`).  

<div align="center">
 <img width="419" height="222" alt="image" src="https://github.com/user-attachments/assets/a75f2441-53af-4fa4-a52b-fe28448a8548" />
</div>  

Among several candidate UV layouts, we **adopted the final design**, which balances the **number of vertices and UV area**.  
This choice ensures that regions with denser vertices are allocated larger UV space, enabling the network to better learn fine-grained details in high-importance areas.  

To obtain **high-resolution meshes (≈3K vertices)**, we interpolate the base MANO UV mapping and transfer correspondences to the upsampled mesh. This interpolation guarantees consistency between the **coarse MANO-level meshes** and the **high-fidelity super-resolution meshes**, while also using the corresponding **high-fidelity hand UVs** as supervision during training.


---

## 📖 Citation

If you find this work useful, please cite:  
 ```
    @inproceedings{chen2021i2uv,
    title={I2uv-handnet: Image-to-uv prediction network for accurate and high-fidelity 3d hand mesh modeling},
    author={Chen, Ping and Chen, Yujin and Yang, Dong and Wu, Fangyin and Li, Qin and Xia, Qingpei and Tan, Yong},
    booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
    pages={12929--12938},
    year={2021}
    }
 ```

 
---

## 🤝 Acknowledgements

This code builds upon prior works on UV-based human modeling and mesh regression.  

Special thanks to the community datasets: FreiHAND, HO3D, ObMan, and YT-3D dataset.  

---

## 📌 Disclaimer

This project is from **ICCV 2021**, and the released code may be incomplete due to archival recovery.  
We share it *as is* to benefit the community and welcome any contributions to improve or extend it.  
