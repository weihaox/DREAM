## R-VAC

Reverse VAC (R-VAC) replicates the opposite operations of the VAC brain region, analogously extracting semantics (in the form of CLIP embedding) from fMRI. 

Please first download "Natural Scenes Dataset" and "COCO_73k_annots_curated.npy" file from [NSD HuggingFace](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main) to `nsd_data`. Then use the following command to run R-VAC training
```
python train_dream_rvac.py --data_path nsd_data
```

We basically adopt the same code as [MindEye](https://medarc-ai.github.io/mindeye/)'s high-level pipeline without the diffusion prior and utilize different dimensions. Specifically, fMRI voxels are processed by the MLP backbone and MLP projector to produce the CLIP fMRI embedding, and are trained with a data-augmented contrastive learning loss. The additional MLP projector helps capture meaningful semantics. We did not employ a Prior. The role of the Prior (regardless of the performance gain in MindEye) appears to be substitutable by the MLP projector with MSE losses. In our work, we focus solely on text embedding and disjointed issues of text and image embeddings are not our concerns.

Empirically, The predicted CLIP-text embedding sometimes would not significantly affect results, with the dominance lying in the depth and color guidances. The reconstructed results from T2I-Adapter, as we mentioned in the ablation study, are not very "stable" and requires mannual adjustments to get pleasing results in certain cases. 