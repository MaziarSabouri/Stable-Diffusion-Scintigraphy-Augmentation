import time
import torch
import os
import pandas as pd
import numpy as np
import logging
import argparse
from PIL import Image
from diffusers import UNet2DConditionModel, AutoPipelineForImage2Image, StableDiffusionPipeline
from diffusers.utils import load_image
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("diffusers").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Fast Stable Diffusion Inference Script")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["image2image", "image_text2image", "mask_text2image", "text2image"],
        help="Generation mode corresponding to previous distinct scripts."
    )
    return parser.parse_args()

def is_black_image(image, threshold=5):
    """
    Checks if the generated image is completely black.
    :param image: PIL image
    :param threshold: Pixel intensity threshold to determine if it's black
    :return: True if black, False otherwise
    """
    grayscale = image.convert("L")  # Convert to grayscale
    extrema = grayscale.getextrema()  # Get min/max pixel values
    return extrema[1] <= threshold  # If max pixel intensity is low, image is black

def main():
    args = parse_args()

    # Configurations Common
    BATCH_SIZE = 256
    NUM_STREAMS = 1
    MAX_RETRIES = 3
    
    # Mode-specific Configurations
    if args.mode == "image2image":
        N_Sample_Per_Target = 1000
        data_path = "../01-Image2Image"
        base_img_dir = "../IMAGES"
        pipeline_class = AutoPipelineForImage2Image
        use_prompt = False
        requires_init_image = True
    elif args.mode == "image_text2image":
        N_Sample_Per_Target = 1000
        data_path = "../02-image_text2image"
        base_img_dir = "../IMAGES"
        pipeline_class = AutoPipelineForImage2Image
        use_prompt = True
        requires_init_image = True
    elif args.mode == "mask_text2image":
        N_Sample_Per_Target = 1000
        data_path = "../03-mask_text2image"
        base_img_dir = "../MASKS"
        pipeline_class = AutoPipelineForImage2Image
        use_prompt = True
        requires_init_image = True
    elif args.mode == "text2image":
        N_Sample_Per_Target = 1000
        data_path = "../04-Text2Image"
        base_img_dir = "../IMAGES"
        pipeline_class = StableDiffusionPipeline
        use_prompt = True
        requires_init_image = False

    # Load model
    model_path = "../Stable Diffusion/Output/checkpoint-15000"
    unet = UNet2DConditionModel.from_pretrained(model_path + "/unet", torch_dtype=torch.float16)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline_class.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.float16)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Prepare Data Paths
    os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)

    # Load dataset
    data_frame = pd.read_excel(
        "../Data/Scintigraphy.xlsx",
        sheet_name="Train",
    )
    targets = np.unique(data_frame.Class3.values)
    
    # Needs a shared dictionary for proper counting across threads (or single-thread management)
    counters = {target: 0 for target in targets}

    # Start the timer
    start_time = time.time()

    # Create CUDA streams for parallel execution
    streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]

    def process_batch(target, batch_samples):
        try:
            init_images = []
            prompts = []
            img_names = []

            target_dir = os.path.join(data_path, str(target))
            os.makedirs(target_dir, exist_ok=True)

            for im in range(len(batch_samples)):
                img_name = batch_samples.Folder.values[im]
                prompt = batch_samples.Prompt.values[im] if use_prompt else ""
                img_path = f"{base_img_dir}/{img_name}.jpg"

                if os.path.exists(img_path):
                    if requires_init_image:
                        init_images.append(load_image(img_path))
                    prompts.append(prompt)
                    img_names.append(img_name)
                else:
                    print(f"⚠️ Warning: Image not found -> {img_path}")

            if len(prompts) == 0:
                print(f"❌ Skipping target {target}: No valid images/prompts found")
                return

            with torch.cuda.stream(streams[target % NUM_STREAMS]):
                with torch.autocast(device_type=device, dtype=torch.float16):
                    if args.mode == "text2image":
                        generated_images = pipeline(prompt=prompts, height=128, width=128).images
                    else:
                        generated_images = pipeline(prompt=prompts, image=init_images).images

                # Retry if black images are detected
                for i, img in enumerate(generated_images):
                    retry_count = 0
                    while is_black_image(img) and retry_count < MAX_RETRIES:
                        print(f"⚠️ Black image detected for {img_names[i]}. Retrying... ({retry_count + 1}/{MAX_RETRIES})")
                        with torch.autocast(device_type=device, dtype=torch.float16):
                            if args.mode == "text2image":
                                img = pipeline(prompt=prompts[i], height=128, width=128).images[0]
                            else:
                                img = pipeline(prompt=prompts[i], image=init_images[i]).images[0]
                        retry_count += 1

                    # Save the final valid image
                    outputImageFileName = os.path.join(target_dir, f"{counters[target]:05d}.jpg")
                    counters[target] += 1
                    img.save(outputImageFileName)

                print(f"✅ Saved {len(generated_images)} images for target {target}")

            del generated_images
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error in batch processing for target {target}: {e}")

    # Parallel Processing
    with ThreadPoolExecutor(max_workers=NUM_STREAMS) as executor:
        future_to_target = {}

        for target in targets:
            df2 = data_frame[data_frame["Class3"] == target]
            df2_sample = df2.sample(n=N_Sample_Per_Target, replace=True)

            target_dir = os.path.join(data_path, str(target))
            os.makedirs(target_dir, exist_ok=True)

            df2_sample.to_excel(os.path.join(data_path, f"{str(target)}.xlsx"))

            batch_list = [df2_sample.iloc[i : i + BATCH_SIZE] for i in range(0, len(df2_sample), BATCH_SIZE)]

            for batch in batch_list:
                future = executor.submit(process_batch, target, batch)
                future_to_target[future] = target

        for future in as_completed(future_to_target):
            target = future_to_target[future]
            try:
                future.result()
            except Exception as exc:
                print(f"❌ Exception in processing target {target}: {exc}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"🔥 Total running time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
