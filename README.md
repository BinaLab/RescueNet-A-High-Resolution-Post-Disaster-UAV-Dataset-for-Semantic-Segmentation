# RescueNet Dataset 

## Overview

Frequent, and increasingly severe, natural disasters threaten human health, infrastructure, and natural systems. The provision of accurate, timely, and understandable information has the potential to revolutionize disaster management. For quick response and recovery on a large scale, after a natural disaster such as a hurricane, access to aerial images is critically important for the response team. The emergence of small unmanned aerial systems (UAS) along with inexpensive sensors presents the opportunity to collect thousands of images after each natural disaster with high flexibility and easy maneuverability for rapid response and recovery.  Moreover, UAS can access hard-to-reach areas and perform data collection  tasks that can be unsafe for humans if not impossible.  Despite all these advancements and efforts to collect such large datasets, analyzing them and extracting meaningful information remains a significant challenge in scientific communities.

[FloodNet](https://ieeexplore.ieee.org/document/9460988) provides high-resolution UAS imageries with detailed semantic annotation regarding the damages.

![alt text](https://github.com/BinaLab/FloodNet-Supervised_v1.0/blob/main/FloodNet-Sample.jpg?raw=true)

## Dataset Details

The data is collected with a small UAS platform, DJI Mavic Pro quadcopters, after Hurricane Harvey. The whole dataset has 2343 images, divided into training (~60%), validation (~20%), and test (~20%) sets. The semantic segmentation labels include: 1) Background, 2) Building Flooded, 3) Building Non-Flooded, 4) Road Flooded, 5) Road Non-Flooded, 6) Water, 7)Tree, 8) Vehicle, 9) Pool, 10) Grass. 

The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD?usp=sharing

### Paper Link
The paper can be downloaded from this [link](https://ieeexplore.ieee.org/document/9460988).
Please cite our paper when using the dataset

 ```
 @ARTICLE{9460988,
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin Roberson},
  journal={IEEE Access}, 
  title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding}, 
  year={2021},
  volume={9},
  number={},
  pages={89644-89654},
  doi={10.1109/ACCESS.2021.3090981}
  }
 
@article{rahnemoonfar2020floodnet,
  title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin},
  journal={arXiv preprint arXiv:2012.02951},
  year={2020}
}

```
