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
  <a href="https://www.youtube.com/watch?v=cUdkeigISOo" target='_blank'>[Video]</a>  •
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
- [2024/04/11] Check out the multimodal decoding method [UMBRAE](https://weihaox.github.io/UMBRAE) and benchmark [BrainHub](https://github.com/weihaox/BrainHub).
- [2023/11/14] Add instructions for R-VAC and R-PKM.
- [2023/10/23] DREAM is accepted by WACV 2024.
- [2023/10/03] Both <a href="https://weihaox.github.io/DREAM">project</a> and <a href="https://arxiv.org/abs/2310.02265">arXiv</a> are available.

## Installation

Please follow the [Installation Instruction](https://github.com/weihaox/DREAM/blob/main/docs/install.md) to setup all the required packages.

## Training DREAM

### R-VAC

R-VAC (Reverse VAC) replicates the opposite operations of the VAC brain region, analogously extracting semantics (in the form of CLIP embedding) from fMRI. 

To train R-VAC, please first download "Natural Scenes Dataset" and "COCO_73k_annots_curated.npy" file from [NSD HuggingFace](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main) to `nsd_data`, then use the following command to run R-VAC training
```
python train_dream_rvac.py --data_path nsd_data
```

We basically adopt the same code as [MindEye](https://medarc-ai.github.io/mindeye/)'s high-level pipeline without the diffusion prior and utilize different dimensions. Specifically, fMRI voxels are processed by the MLP backbone and MLP projector to produce the CLIP fMRI embedding, and are trained with a data-augmented contrastive learning loss. The additional MLP projector helps capture meaningful semantics. We did not employ a Prior. The role of the Prior (regardless of the performance gain in MindEye) appears to be substitutable by the MLP projector with MSE losses. In our work, we focus solely on text embedding and disjointed issues of text and image embeddings are not our concerns.

Empirically, The predicted CLIP-text embedding sometimes would not significantly affect results, with the dominance lying in the depth and color guidances. The reconstructed results from T2I-Adapter, as we mentioned in the ablation study, are not very "stable" and requires mannual adjustments to get pleasing results in certain cases. 

### R-PKM 

R-PKM (Reverse PKM) maps fMRI to color and depth in the form of spatial palettes and depth maps.

#### Data Acquisition and Preprocessing

For the training of the encoder in stage 1 and the decoder in stage 2, we use MiDaS-estimated depth maps as the surrogate ground-truth depth. Please process the NSD images accordingly.

For the training of the decoder in stage 3, please process additional natural images from sources like [ImageNet](https://www.image-net.org/), [LAION](https://laion.ai/), or others to obtain their estimated depth maps.

#### Training and Testing

The architectures of Encoder and Decoder are built on top of [SelfSuperReconst](https://github.com/WeizmannVision/SelfSuperReconst), with inspirations drawn from [VDVAE](https://github.com/openai/vdvae). The Encoder training is implemented in `train_encoder.py` and the Decoder training is implemented in `train_decoder.py`. 

Train RGB-only Encoder (supervised-only):
```bash
python $(scripts/train_enc_rgbd.sh)
```
Then train RGB-only Decoder (supervised + self-supervised):
```bash
python $(scripts/train_dec_rgbd.sh)
```
Please refer to their implementations for further details.

If you don't have enough resources for training the models, an alternative computationally-efficient way is to use regression models. Specifically, you can use a similar way employed in [StableDiffusionReconstruction](https://github.com/yu-takagi/StableDiffusionReconstruction) or [brain-diffuser](https://github.com/ozcelikfu/brain-diffuser) to first extract latent features of stimuli images and depth estimations from [VDVAE](https://github.com/openai/vdvae) for any subject 'x' and then learn regression models from fMRI to VDVAE latent features and save test predictions. The final RGBD outcomes can be reconstructed from the predicted test features. 

The color information deciphered from the fMRI data is in the form of spatial palettes. These spatial palettes can be obtained by first downsampling (with bicubic interpolation) a predicted image and then upsampling (with nearest interpolation) it back to its original resolution, as shown above.

```python
def get_cond_color(cond_image, mask_size=64):
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.BICUBIC)
    color = cond_image.resize((H, W), Image.NEAREST)
    return color
```

The predicted RGBD are then used to facilitate subsequent image reconstruction by the Color Adapter (C-A) and the Depth Adapter (D-A) in T2I-Adapter in conjunction with SD. Depite coarse results, the predicted depth is sufficient in most cases to guide the scene structure and object position such as determining the location of an airplane or the orientation of a bird standing on a branch. Similarly, despite not precisely preserving the local color, the estimated color palette provide a reliable constraint and guidance on the overall scene appearance. Below are some examples.


## Reconstructing from pre-trained DREAM (GIR)

This process is done with the color and depth adapters in the CoAdapter (Composable Adapter) mode of [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter). There are instances where manual adjustments to the weighting parameters become necessary to achieve images of the desired quality, semantics, structure, and appearance. You may find helpful guidance on how to achieve satisfactory results with the CoAdapter in the [tips](https://github.com/TencentARC/T2I-Adapter/blob/SD/docs/coadapter.md#useful-tips) section.


## Evaluation

The scripts for assessing reconstructed images using the identical low- and high-level image metrics employed in the paper are available on [Brain-Diffuser](https://github.com/ozcelikfu/brain-diffuser) (recommended if using images) and [MindEye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD). 

Regarding the evaluation of depth and color, we use metrics from depth estimation and color correction to assess depth and color consistencies in the reconstructed images. The depth metrics include Abs Rel (absolute error), Sq Rel (squared error), RMSE (root mean squared error), and RMSE log (root mean squared logarithmic error). For color assessment, we use CD (Color Discrepancy) and STRESS (Standardized Residual Sum of Squares). Please consult the [supplementary material](https://arxiv.org/pdf/2310.02265.pdf) for details.

For depth, we first use a depth estimation method (specifically, [MiDaS](https://github.com/isl-org/MiDaS)) to obtain depth maps for both reconstructed and ground-truth images. We then derive quantitative results by running the following:
```bash
python src/evaluation/cal_depth_metric.py --method ${METHOD_NAME}  --save_csv ${SAVE_CSV_OR_NOT} \
    --mode ${DEPTH_AS_UNIT8_OR_FLOAT32} \
    --all_images_path ${PATH_TO_GT_IMAGE} --recon_path ${PATH_TO_RECONSTRUCTED_IMAGE} 
``` 

For color, we utilize the following script to obtain the CD and STRESS results:
```bash
python src/evaluation/cal_color_metric.py --method ${METHOD_NAME} --save_csv ${SAVE_CSV_OR_NOT} \
    --all_images_path ${PATH_TO_GT_IMAGE} --recon_path ${PATH_TO_RECONSTRUCTED_IMAGE} 
``` 
We have also included an additional [color relevance](https://github.com/weihaox/DREAM/blob/main/src/evaluation/cal_color_metric.py#L111) metric (CD) based on Hellinger distance.

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