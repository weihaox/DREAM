<div align="center">

<h2>DREAM: Visual Decoding from Reversing Human Visual System</h2>

<div>
    <a href='https://weihaox.github.io/' target='_blank'>Weihao Xia</a><sup>1</sup>&emsp;
    <a href='https://team.inria.fr/rits/membres/raoul-de-charette/' target='_blank'>Raoul de Charette</a><sup>2</sup>&emsp;
    <a href='https://www.cl.cam.ac.uk/~aco41/' target='_blank'>Cengiz Öztireli</a><sup>3</sup>&emsp;
    <a href='http://www.homepages.ucl.ac.uk/~ucakjxu/' target='_blank'>Jing-Hao Xue</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>University College London&emsp;
    <sup>2</sup>Inria&emsp;
    <sup>3</sup>University of Cambridge&emsp;
</div>
<h3 align="center">WACV 2024</h3>

---

<h4 align="center">
  <a href="https://weihaox.github.io/DREAM" target='_blank'>[Project Page]</a> •
  <a href="https://arxiv.org/pdf/2310.02265" target='_blank'>[arXiv]</a> <br> <br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=weihaox/DREAM" width="8%" alt="visitor badge"/>
</h4>

</div>

>**Abstract:** In this work we introduce DREAM: Visual Decoding from REversing HumAn Visual SysteM. DREAM represents an fMRI-to-image method designed to  reconstruct visual stimuli based on brain activities, grounded on fundamental knowledge of the human visual system (HVS). By crafting reverse pathways, we emulate the hierarchical and parallel structures through which humans perceive visuals. These specific pathways are optimized to decode semantics, color, and depth cues from fMRI data, mirroring the forward pathways from the visual stimuli to fMRI recordings. To achieve this, we have implemented two components that mimic the reverse processes of the HVS.  The first, the Reverse Visual Association Cortex (R-VAC, Semantics Decipher), retraces the pathways of this particular brain region, extracting semantics directly from fMRI data. The second, the Reverse Parallel PKM (R-PKM, Depth & Color Decipher), predicts both color and depth cues from fMRI data concurrently. The Guided Image Reconstruction (GIR) is responsible for reconstructing final images from deciphered semantics, color, and depth cues by using the Color Adapter (C-A) and the Depth Adapter (D-A) in T2I-Adapter in conjunction with Stable Diffusion (SD).

<div align="center">
<tr>
    <img src="docs/images/method_overview.png" width="90%"/>
</tr>
</div>

## News :triangular_flag_on_post:

- [2022/10/23] DREAM is accepted by WACV 2024.
- [2023/10/03] Both <a href="https://weihaox.github.io/DREAM">project</a> and <a href="https://arxiv.org/abs/2310.02265">arXiv</a> are available.

## Installation

Please follow the [Installation Instruction](https://github.com/weihaox/DREAM/blob/main/docs/install.md) to setup all the required packages.

## Training DREAM

### R-VAC

Reverse VAC replicates the opposite operations of the VAC brain region, analogously extracting semantics (in the form of CLIP embedding) from fMRI.

> TBD

### R-PKM 

Reverse PKM maps fMRI to color and depth in the form of spatial palettes and depth maps to facilitate subsequent processing by the Color Adapter (C-A) and the Depth Adapter (D-A) in T2I-Adapter in conjunction with SD for image reconstruction from deciphered semantics, color, and depth cues.

> TBD

## Reconstructing from pre-trained DREAM (GIR)

This process is done with the color and depth adapters in the CoAdapter (Composable Adapter) mode of [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter). There are instances where manual adjustments to the weighting parameters become necessary to achieve images of the desired quality, semantics, structure, and appearance. You may find helpful guidance on how to achieve satisfactory results with the CoAdapter in the [tips](https://github.com/TencentARC/T2I-Adapter/blob/SD/docs/coadapter.md#useful-tips) section.


## Evaluation

The scripts for assessing reconstructed images using the identical low- and high-level image metrics employed in the paper are available on [Brain-Diffuser](https://github.com/ozcelikfu/brain-diffuser) (recommended if using images) and [MindEye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD). 

Regarding the evaluation of depth and color, we use metrics from depth estimation and color correction to assess depth and color consistencies in the reconstructed images. The depth metrics include Abs Rel (absolute error), Sq Rel (squared error), RMSE (root mean squared error), and RMSE log (root mean squared logarithmic error). For color assessment, we use CD (Color Discrepancy) and STRESS (Standardized Residual Sum of Squares). Please consult the [supplementary material](https://arxiv.org/pdf/2310.02265.pdf) for details.

For depth assessment, we first use a depth estimation method (specifically, MiDaS) to obtain the depth maps for both reconstructed and ground-truth image. We then derive quantitative results by running the following:
```bash
python src/evaluation/cal_depth_metric.py --method ${METHOD_NAME}  --save_csv ${SAVE_CSV_OR_NOT} \
    --mode ${DEPTH_AS_UNIT8_OR_FLOAT32} \
    --all_images_path ${PATH_TO_GT_IMAGE} --recon_path ${PATH_TO_RECONSTRUCTED_IMAGE} 
``` 

For evaluation, we utilize the following script to obtain the CD and STRESS results:
```bash
python src/evaluation/cal_color_metric.py --method ${METHOD_NAME} --save_csv ${SAVE_CSV_OR_NOT} \
    --all_images_path ${PATH_TO_GT_IMAGE} --recon_path ${PATH_TO_RECONSTRUCTED_IMAGE} 
``` 

## References
- Codes in R-VAC are based on [MedARC-AI/fMRI-reconstruction-NSD](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
- Codes in R-RKM draw inspiration from [WeizmannVision/SelfSuperReconst](https://github.com/WeizmannVision/SelfSuperReconst)
- Codes in GIR are derived from earlier version of [TencentARC/T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)
- Dataset used in the studies are obtained from [Natural Scenes Dataset](https://naturalscenesdataset.org/)

## Citation

```bibtex
@inproceedings{xia2023dream,
  author    = {Xia, Weihao and de Charette, Raoul and Öztireli, Cengiz and Xue, Jing-Hao},
  title     = {DREAM: Visual Decoding from Reversing Human Visual System},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2024},
}
```