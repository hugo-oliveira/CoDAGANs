# CoDAGANs

This repository contains the official implementation for <a href="https://arxiv.org/abs/1901.05553">Conditional Domain Adaptation Generative Adversarial Networks (CoDAGANs)</a>. CoDAGANs allow for multi-dataset Unsupervised, Semi-Supervised and Fully Supervised Domain Adaptation (UDA, SSDA and FSDA) between Biomedical Image datasets with distinct visual features due to different digitization procedures/equipment.


<img src="https://github.com/hugo-oliveira/CoDAGANs/blob/master/misc/CoDAGAN_Architecture.png" alt="CoDAGAN Overview">

If you have any doubt regarding the paper, methodology or code, please contact oliveirahugo [at] dcc.ufmg.br and/or jefersson [at] dcc.ufmg.br.

## Training.
For training a CoDAGAN from scratch using config file CXR_lungs_MUNIT_1.0.yaml:
```
python train.py --config configs/CXR_lungs_MUNIT_1.0.yaml
```

## Testing.
For testing epoch 400 of a pretrained CoDAGAN with configs CXR_lungs_MUNIT_1.0.yaml:
```
python test.py --load 400 --snapshot_dir outputs/CXR_lungs_MUNIT_1.0/checkpoints/ --config configs/CXR_lungs_MUNIT_1.0.yaml
```

## Acknowledgment
Authors would like to thank NVIDIA for the donation of the GPUs and for the financial support provided by CAPES, CNPq and FAPEMIG (APQ-00449-17) that allowed the execution of this work
