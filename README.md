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

---

<h4 align="center">
  <a href="https://weihaox.github.io/DREAM" target='_blank'>[Project Page]</a> •
  <a href="" target='_blank'>[arXiv]</a> <br> <br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=weihaox/DREAM" width="8%" alt="visitor badge"/>
</h4>

</div>

>**Abstract:** In this work we introduce DREAM: Visual Decoding from REversing HumAn Visual SysteM. DREAM represents an fMRI-to-image method designed to  reconstruct visual stimuli based on brain activities, grounded on fundamental knowledge of the human visual system (HVS). By crafting reverse pathways, we emulate the hierarchical and parallel structures through which humans perceive visuals. These specific pathways are optimized to decode semantics, color, and depth cues from fMRI data, mirroring the forward pathways from the visual stimuli to fMRI recordings. To achieve this, we have implemented two components that mimic the reverse processes of the HVS.  The first, the Reverse Visual Association Cortex (R-VAC), retraces the pathways of this particular brain region, extracting semantics directly from fMRI data. The second, the Reverse Parallel PKM (R-PKM), predicts both color and depth cues from fMRI data concurrently. The final images are reconstructed from deciphered semantics, color, and depth cues by using the Color Adapter (C-A) and the Depth Adapter (D-A) in T2I-Adapter in conjunction with SD.

<div align="center">
<tr>
    <img src="docs/images/method_overview.png" width="90%"/>
</tr>
</div>

The code will be available soon.