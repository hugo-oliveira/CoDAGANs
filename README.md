# CoDAGANs

This repository contains the official implementation for <a href="http://www.patreo.dcc.ufmg.br/codagans/">Conditional Domain Adaptation Generative Adversarial Networks (CoDAGANs)</a>. CoDAGANs allow for multi-dataset Unsupervised, Semi-Supervised and Fully Supervised Domain Adaptation (UDA, SSDA and FSDA) between Biomedical Image datasets with distinct visual features due to different digitization procedures/equipment.


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
