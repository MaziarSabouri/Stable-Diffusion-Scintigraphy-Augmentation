# Stable Diffusion Inference for Thyroid Scintigraphy Augmentation

This repository contains the inference scripts used to generate synthetic thyroid scintigraphy images via a fine-tuned Stable Diffusion model. This code supports the data augmentation pipeline presented in our published paper.

To streamline usage, the original four individual inference scripts have been combined into a unified, high-performance script (`Inference_combined.py`) that utilizes CUDA streams for parallel processing and includes automatic handling of generated black images (safety checker retries).

## 🚀 Features

- **Four Generation Modes**: Unconditional image-to-image, text-guided image-to-image, mask-guided generation, and pure text-to-image.
- **Optimized for Speed**: Utilizes FP16 precision and CUDA streams for efficient batch processing.
- **Robust Generation**: Automatically detects collapsed/black output images and regenerates them up to a specified retry limit.
- **Dataset Integration**: Seamlessly reads from an Excel dataset registry containing metadata, class labels, and text prompts.

## 🛠️ Prerequisites & Installation

Ensure you have a CUDA-capable GPU and the necessary drivers installed. Use python 3.8+ and install the dependencies:

```bash
pip install -r requirements.txt
```

*Note: The key libraries used include `torch`, `diffusers`, `transformers`, `pandas`, and `Pillow`.*

## 📂 Data Preparation

Before running the inference script, ensure your directories and files are set up properly in the source code. You will need:
1. **Fine-tuned Checkpoint**: A Hugging Face style `UNet` and Stable Diffusion v1-4 checkpoint directory.
2. **Metadata Excel File**: containing `Folder` (image ID), `Prompt` (text condition), and `Class3` (target class).
3. **Image Directory**: Cropped/Processed source images (e.g., `Train_crop3`).
4. **Mask Directory**: Label maps used for spatial conditioning (e.g., `Label-JPG`).

*Make sure to update the path variables (`model_path`, `data_frame`, and `base_img_dir`) inside `Inference_combined.py` if your local environment differs from the defaults.*

## 💻 Usage

We provide a single entry-point script. You must specify the `--mode` argument corresponding to the augmentation strategy you wish to employ:

```bash
python Inference_combined.py --mode <GENERATION_MODE>
```

### Supported Modes:

#### 1. Image to Image (`--mode image2image`)
Generates variations of the source image without relying on textual prompts (empty prompt). Ideal for producing pure variations tied intimately to the input distribution.
* **Pipeline**: `AutoPipelineForImage2Image`
* **Input**: Source images + empty prompt

#### 2. Image & Prompt to Image (`--mode image_text2image`)
Guides the variation of the source image using specific text prompts detailing the condition or structure.
* **Pipeline**: `AutoPipelineForImage2Image`
* **Input**: Source images + Text Prompts

#### 3. Mask & Prompt to Image (`--mode mask_text2image`)
Uses segmentation masks (labels) instead of raw RGB images as the spatial basis. The structure of the mask guides the layout, while the text prompt dictates the semantic rendering of the scintigraphy.
* **Pipeline**: `AutoPipelineForImage2Image`
* **Input**: Label Masks + Text Prompts

#### 4. Text to Image (`--mode text2image`)
Pure generative mode starting from standard latent noise, fully directed by the text prompts mapping domain knowledge into visual outputs.
* **Pipeline**: `StableDiffusionPipeline`
* **Input**: Text Prompts only

## 📊 Outputs

Depending on the mode used, the output images are distributed into target-specific folders (e.g., `0/`, `1/`, etc.) located within their respective main directories (`01-Image2Image/`, `02-image_text2image/`, etc.). 

The code also exports a sampling Excel file (`<ClassID>.xlsx`) into the target directory to record the exact metadata of the sampled instances.

## 📝 Citation

If you use this code in your research, please consider citing our published work (placeholder):

```bibtex
@article{thyroid_scintigraphy_augmentation,
  title={Thyroid Scintigraphy Data Augmentation using Fine-tuned Stable Diffusion},
  author={[Your Name / Authors]},
  journal={[Journal Name / Venue]},
  year={[Year]},
  publisher={[Publisher]}
}
```

## 📄 License
This project is released under the [Apache License 2.0](LICENSE) / appropriate academic license.
