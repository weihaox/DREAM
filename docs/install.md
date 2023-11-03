## Environment Setup

1. Run `setup.sh` to create a conda environment and activate the environment with conda activate.
```sh
# install libraries and activate the conda environment
cd DREAM/src
. setup.sh
conda activate dream
```
2. Config the T2I-Adapter and Stable Diffusion. Please refer to [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) to download the pretrained models of the color adapter, depth adapter, and Stable Diffusion (SD-V1.4/1.5).

3. Data Acquisition and Processing. Complete the Natural Scenes Dataset (NSD) [Data Access Form](https://forms.gle/xue2bCdM9LaFNMeb7) and consent to the [Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions). Once completed, you will (immediately) receive an e-mail with a link to the NSD Data Manual that contains further instructions on how to access and download the data. 