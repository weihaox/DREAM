## R-PKM

Reverse PKM (R-PKM) maps fMRI to color and depth in the form of spatial palettes and depth maps to facilitate subsequent image reconstruction by the Color Adapter (C-A) and the Depth Adapter (D-A) in T2I-Adapter in conjunction with SD. Depite coarse results, the predicted depth is sufficient in most cases to guide the scene structure and object position such as determining the location of an airplane or the orientation of a bird standing on a branch. Similarly, despite not precisely preserving the local color, the estimated color palette provide a reliable constraint and guidance on the overall scene appearance. Below are some examples.

<div align="center">
<tr>
    <img src="docs/images/RPKM_result.png" width="90%"/>
</tr>
</div>

### Data Preprocessing

For the training of the encoder in stage 1 and the decoder in stage 2, we use MiDaS-estimated depth maps as the surrogate ground-truth depth. Please process the NSD images accordingly.

For the training of the decoder in stage 3, please process additional natural images from sources like [ImageNet](https://www.image-net.org/), [LAION](https://laion.ai/), or others to obtain their estimated depth maps.

### Code Structure

The architectures of Encoder and Decoder are built on top of [SelfSuperReconst](https://github.com/WeizmannVision/SelfSuperReconst), with inspirations drawn from [VDVAE](https://github.com/openai/vdvae). Please refer to their implementations for further details.

The color information deciphered from the fMRI data is in the form of spatial palettes. These spatial palettes can be obtained by first downsampling (with bicubic interpolation) a predicted image and then upsampling (with nearest interpolation) it back to its original resolution, as shown above.

```python
def get_cond_color(cond_image, mask_size=64):
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.BICUBIC)
    color = cond_image.resize((H, W), Image.NEAREST)
    return color

def get_cond_color(cond_image, mask_size=64):
    H, W = cond_image.shape[:2]  
    color = cv2.resize(cond_image, (W // mask_size, H // mask_size), interpolation=cv2.INTER_CUBIC)
    color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
    return color
```