<div align='center'>
    <b><font size='30'> GenCLS</font></b>
</div>

## Introduction 
GenCLS is an open source image classification toolbox based on PyTorch. 

### Majoir features

- Various backbones 
- Bag of training tricks
- Intergrate model compression: Prunning, Quantization, KD
- Intergrate model export format: ONNX, TensorRT 

##  Installation 

## Data 


## Get start 
- Training from scratch:
```bash
python tools/train.py --config <config_file> --exp_dir <exp_dir>
python tools/train.py --config config/print_hw_cls.yml --exp_dir experiments/exp_1
```
- Train from pretrained model:
```bash 
python tools/train.py --config <config_file> --exp_dir <exp_dir> --pretrained <pretrained_model>
```

- Resume training from checkpoint:
```bash
python tools/train.py --config <config_file> --exp_dir <exp_dir> --resume_from <checkpoint>
```

## Model Zoo 
- [x] MobileNetV3 
- [ ] ResNet

## Experiment with printed-handwriting classification
- [ ] MobileNetV3_large_scale_1
- [ ] MobileNetV3_large_scale_0.75
- [ ] MobileNetV3_small_scale_0.75
- [ ] MobileNetV3_small_scale_0.35

### Aug
- [ ] Custom Augment: Color + Distort
- [ ] RandAugment

###  Benchmark models

## TODO 
- [x] Training module 
    - [ ] Cache dataset for faster training
- [ ] Optimize config 
- [ ] Add trainin tricks
    - [ ] AMP 
    - [ ] EMA 
- [ ] Benchmark model
- [ ] Model compression 
- [ ] Model export 
