# [ECCV 2026] 4D-VGGT: A SpatioTemporal Foundation Model for Dynamic Scene Geometry Estimation

[Haonan Wang](https://scholar.google.com.hk/citations?hl=zh-CN&view_op=list_works&gmla=AH8HC4wel7f5UzHZm3NN_RHl9by4ODKcg12HuynxhWBbyyFpY3GCQp_wRryBPNSci76ZfoOB8_IDasu-vEEyzy9skm3tDy0&user=LCNXgmAAAAAJ) $^{1}$, [Hanyu Zhou](https://hyzhouboy.github.io/) $^{2✉}$,  [Haoyue Liu](https://scholar.google.com.hk/citations?hl=zh-CN&user=DadbHdAAAAAJ) $^1$, [Luxin Yan](https://scholar.google.com.hk/citations?user=5CS6T8AAAAAJ&hl=zh-CN) $^{1}$

$^1$ Huazhong University of Science and Technology  $^2$ National University of Singapore

$^✉$ Corresponding Author.

## Overview

![fig1](./images/Figure_1.png)

![fig2](./images/Figure_2.png)

4D-VGGT mainly contains two parts: three main components: 1) **Multi-Setting Input.** We design an adaptive visual grid that enables our model to accommodate visual features from diverse camera configurations through attention masks. 2) **Multi-Level Representation.** We propose a cross-view global fusion module to learn the spatial representation between various views, and a cross-time local fusion to model the temporal representation along continuous time steps. 3) **Multi-Task Prediction.** We construct multiple task-specific heads and perform joint multi-task optimization to learn the corresponding spatiotemporal features for scene geometry. Under our unified framework, these components enable our model to utilize a shared spatiotemporal representation scheme to support diverse input configurations and accomplish a variety of visual tasks.

## News

2026.06.20: Our paper is accepted by ECCV 2026. 



## Citation

If you find this repository/work helpful in your research, welcome to cite our paper and give a ⭐.

```
@article{wang20254d,
  title={4D-VGGT: A General Foundation Model with SpatioTemporal Awareness for Dynamic Scene Geometry Estimation},
  author={Wang, Haonan and Zhou, Hanyu and Liu, Haoyue and Yan, Luxin},
  journal={arXiv preprint arXiv:2511.18416},
  year={2025}
}
```

