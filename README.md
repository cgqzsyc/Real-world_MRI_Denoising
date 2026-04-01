<div align="center">

<h1><a href="https://doi.org/10.1038/s41746-026-02548-y">[npj Digital Medicine 2026] Real-world Unified Denoising for Multi-organ Fast MRI: A Large-scale Prospective Validation</a></h1>

[Yuchen Shao\*](https://scholar.google.com/citations?user=R0WGScwAAAAJ&hl=zh-CN&oi=ao), Hongyan Huang\*, Lingyan Zhang\*, Dongsheng Li\*, Zhiguang Ding, Fan Wang, Shengli Chen, Shiwei Lin, Yuning Gu, Mu Du, Hongbing Li, Jiuping Liang, Xiaoqian Huang, Aowen Liu, Jiafu Zhong, [Yiqiang Zhan](https://scholar.google.com/citations?user=N4sca6EAAAAJ&hl=zh-CN&oi=ao), [Xiang Sean Zhou](https://scholar.google.com/citations?user=-bp44DoAAAAJ&hl=zh-CN), [Feng Shi](https://scholar.google.com/citations?user=MztESj8AAAAJ&hl=zh-CN&oi=ao), [Shu Liao](https://scholar.google.com/citations?user=INL-unYAAAAJ&hl=zh-CN&oi=ao), Kaicong Sun†, [Dinggang Shen†](https://scholar.google.com/citations?user=v6VYQC8AAAAJ&hl=zh-CN&oi=sra), and Yingwei Qiu†

\* Co-first author  † Corresponding author

</div>

## :fire: Updates

- 2026.03.19 The paper has been published.
- 2026.03.04 The paper has been accepted by _npj Digital Medicine_.

## :parrot: Introduction
In this paper, we proposed a unified denoising model for accelerated MRI, especially for older-generation 1.5T MR scanners. This model can be directly applied to the reconstructed images by commercial reconstruction algorithms embedded in scanners. This study includes:

**Large-scale and diverse real-world MR noisy-clean paired image dataset**
We collected a large-scale prospective dataset, which consists of 5,366 real-world noisy-clean volume pairs (N = 102,060 slice pairs), covering six organs including head (N = 37,482), knee (N = 8,329), C-spine (N = 14,097), L-spine (N = 14,447), T-spine (N = 18,139), and shoulder (N = 9,566) with 82 MRI protocols (e.g., T1w, T1-FLAIR, T2w, T2-FLAIR, DWI, PDw, DIXON), and three MRI manufacturers (i.e., SIEMENS, GE, Philips) acquired from January 2024 to August 2024 in three hospitals in Shenzhen and Guangzhou, China. Besides, for external evaluation, we further collected 2,157 volume pairs (N = 46,870 slice pairs) of healthy and non-healthy subjects including MRI scanners of Siemens, GE, UIH, and Philips from four hospitals in China from October 2024 to March 2025 covering totally 96 MRI protocols (Siemens: 29, GE: 25, UIH: 14, Philips: 19).

![alt text](images/Fig1(a).png)

**Model architecture**
Our denoising framework contains two main modules, namely, a non-linear cascaded-trained degradation model and a text-guided variational diffusion model. These two modules were trained individually and used collaboratively in the inference phase.

![alt text](images/Fig1(b).png)

**Extensive clinical evaluation**
We conducted extensive clinical evaluation to evaluated the real-world clinical performance of our method. Reader studies were conducted by two radiologists with 5 and 3 years of specialty experience, respectively, to evaluate the
clinical utility of our model.

![alt text](images/Fig1(c).png)

# :page_facing_up: Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{shao2026realworldmridenoising,
      title={Real-world Unified Denoising for Multi-organ Fast MRI: A Large-scale Prospective Validation}, 
      author={Yuchen Shao and Hongyan Huang and Lingyan Zhang and Dongsheng Li and Zhiguang Ding and Fan Wang and Shengli Chen and Shiwei Lin and Yuning Gu and Mu Du and Hongbing Li and Jiuping Liang and Xiaoqian Huang and Aowen Liu and Jiafu Zhong and Yiqiang Zhan and Xiang Sean Zhou and Feng Shi and Shu Liao and Kaicong Sun and Dinggang Shen and Yingwei Qiu},
      journal={npj Digital Medicine}
      year={2026}
      doi={https://doi.org/10.1038/s41746-026-02548-y}
}
```

# :dizzy: Acknowledgement

Thanks to the open source of the following projects:

- [NVlabs/RED-diff](https://github.com/NVlabs/RED-diff)
- [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [phillipi/pix2pix](https://github.com/phillipi/pix2pix)
