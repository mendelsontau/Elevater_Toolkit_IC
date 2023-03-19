############## Configuration section begins ##################

# Model Config: [vitb32_CLIP, vitb16_CLIP, mae_vitb16, mocov3_vitb16, vit_base_patch16_224, vit_base_patch32_224, deit_base_patch16_224]
model_cfg=clip_blip

# Mode: [linear_probe, finetune, zeroshot]
mode=zeroshot

# Use FP32 [default: True]
use_fp32=True

# Dataset: [caltech101]
declare -a  datasets=(caltech101 cifar10 cifar100 country211 dtd eurosat-clip fer2013 fgvc-aircraft-2013b flower102 food101 gtsrb hateful-memes imagenet-1k kitti-distance minst oxford-iiit-pets patchcamelyon rendered-sst2 resisc45-clip stanfordcar voc2007classification)
output_dir=./

for dataset in ${datasets[@]}
do
    python vision_benchmark/commands/zeroshot.py --ds vision_benchmark/resources/datasets/$dataset.yaml --model vision_benchmark/resources/model/$model_cfg.yaml MODEL.CLIP_FP32 $use_fp32 DATASET.ROOT $output_dir/datasets 
done