## R-VAC

Reverse VAC (R-VAC) replicates the opposite operations of the VAC brain region, analogously extracting semantics (in the form of CLIP embedding) from fMRI. 

We basically adopt the same code as [MindEye](https://medarc-ai.github.io/mindeye/)'s high-level pipeline without the diffusion prior and utilize different dimensions. Specifically, fMRI voxels are processed by the MLP backbone and MLP projector to produce the CLIP fMRI embedding, and are trained with a data-augmented contrastive learning loss. The additional MLP projector helps capture meaningful semantics. We did not employ a Prior. The role of the Prior (regardless of the performance gain in MindEye) appears to be substitutable by the MLP projector with MSE losses. In our work, we focus solely on text embedding and disjointed issues of text and image embeddings are not our concerns.

To run R-VAC, please implement the following minimal changes to the MindEye training [script](https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/Train_MindEye.py).

- download MindEye:
```bash
git clone https://github.com/MedARC-AI/fMRI-reconstruction-NSD
```
- disable prior: set `--prior` as `False`.
- modify losses: add text similarity in `v2c` ([loss_nce_sum](https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/Train_MindEye.py#L634))

```python
# line 601
clip_target_i = clip_extractor.embed_image(image).float()
clip_target_t = clip_extractor.embed_text(text).float()  

# line 614
clip_target_i_norm = nn.functional.normalize(clip_target_i.flatten(1), dim=-1)
clip_target_t_norm = nn.functional.normalize(clip_target_t.flatten(1), dim=-1)

# line 617
loss_nce_i = utils.mixco_nce(
    clip_voxels_norm,
    clip_target_i_norm, # add clip_target_i_norm and clip_target_i
    temp=.006, 
    perm=perm, betas=betas, select=select)

loss_nce_t = utils.mixco_nce(
    clip_voxels_norm,
    clip_target_t_norm, # add clip_target_t_norm and clip_target_t
    temp=.006, 
    perm=perm, betas=betas, select=select)

# line 630
if prior and v2c:
	# omit code here
elif v2c:
	loss_nce = loss_nce_t + loss_nce_i
    loss_nce_sum += loss_nce.item()
    loss = loss_nce
```

**Notes**: 

The reconstructed results from T2I-Adapter, as we mentioned in the ablation study, are not very "stable" and requires mannual adjustments to get pleasing results. It seems that sometimes the predicted CLIP-text embedding would not significantly affect the results, with the dominance lying in the depth and color guidances.