# RescueNet Dataset 

## Overview

Frequent, and increasingly severe, natural disasters threaten human health, infrastructure, and natural systems. The provision of accurate, timely, and understandable information has the potential to revolutionize disaster management. For quick response and recovery on a large scale, after a natural disaster such as a hurricane, access to aerial images is critically important for the response team. The emergence of small unmanned aerial systems (UAS) along with inexpensive sensors presents the opportunity to collect thousands of images after each natural disaster with high flexibility and easy maneuverability for rapid response and recovery.  Moreover, UAS can access hard-to-reach areas and perform data collection  tasks that can be unsafe for humans if not impossible.  Despite all these advancements and efforts to collect such large datasets, analyzing them and extracting meaningful information remains a significant challenge in scientific communities.

RescueNet provides high-resolution UAS imageries with detailed semantic annotation regarding the damages.

![alt text](https://github.com/tashnimchowdhury/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/blob/main/example-rescuenet-all-cls.PNG?raw=true)


## Dataset Details

The data is collected with a small UAS platform, DJI Mavic Pro quadcopters, after Hurricane Michael. The whole dataset has 4494 images, divided into training (~80%), validation (~10%), and test (~10%) sets. The semantic segmentation labels include: 1) Background, 2) Water, 3)Building No Damage, 4) Building Minor Damage, 5) Building Major Damage, 6) Buidling Total Destruction, 7) Road-Clear, 8) Road-Blocked, 9)Vehicle, 10) Tree, 11) Pool.

 <!--  
The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1XNgPVmiu9egr1fywgNeXfnxojFOe_INT?usp=sharing

-->

The dataset can be downloaded from: [Dropbox](https://www.dropbox.com/scl/fo/ntgeyhxe2mzd2wuh7he7x/AHJ-cNzQL-Eu04HS6bvBgcw?rlkey=6vxiaqve9gp6vzvzh3t5mz0vv&e=1&dl=0) or [figshare](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1)

## License

This dataset is released under the [Creative Common License CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### Paper Link

The paper can be downloaded from this [link](https://www.nature.com/articles/s41597-023-02799-4).
Please cite our paper when using the dataset

 ```
 
@article{rahnemoonfar2023rescuenet,
  title={RescueNet: a high resolution UAV semantic segmentation dataset for natural disaster damage assessment},
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim and Murphy, Robin},
  journal={Scientific data},
  volume={10},
  number={1},
  pages={913},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

```

