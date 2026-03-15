export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/home/msabouri/Projects/01-Thyroid_Scintigraphy_Augmentation/Classification/DDPM/ThyroiDeep/data/Thyroid Dataset/Train_crop3"

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="$TRAIN_DIR" \
  --use_ema \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --resume_from_checkpoint="latest" \
  --max_train_steps=5 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/home/msabouri/Projects/01-Thyroid_Scintigraphy_Augmentation/Scripts/Stable diffusion/Output"
