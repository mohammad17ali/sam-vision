# SAM3-3 Exploration

Exploration and experimentation with Meta's SAM3 (Segment Anything Model 3) - a unified foundation model for promptable segmentation in images and videos.

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (required)
- **Python**: 3.8+
- **VRAM**: 12GB+ recommended

## Installation

```bash
# Activate your UV virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## HuggingFace Authentication

SAM3 is a **gated model** - you need to authenticate with HuggingFace:

### Option 1: Use HuggingFace CLI (Recommended)

```bash
huggingface-cli login
```

Then enter your HuggingFace token (get it from https://huggingface.co/settings/tokens)

### Option 2: Use environment variable

```bash
export HF_TOKEN="your_huggingface_token"
```

### Option 3: SSH Key

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add public key to HuggingFace (https://huggingface.co/settings/keys)
# Then authenticate
huggingface-cli login --token $HF_TOKEN
```

**Important**: Make sure to accept the SAM3 model license on HuggingFace:
1. Visit https://huggingface.co/facebook/sam3
2. Click "Access repository"
3. Accept the license agreement

## What is SAM3?

SAM3 is Meta's latest foundation model for:
- **Image Segmentation**: Text and visual prompts (points, boxes, masks)
- **Video Segmentation**: Object detection, tracking, and segmentation across frames
- **Open-Vocabulary Segmentation**: Segment 270K+ unique concepts with 75-80% human performance

## Usage

```bash
python explore_sam3.py
```

The exploration script demonstrates:
1. Text-based segmentation - describe objects in natural language
2. Multiple prompts - segment different concepts in the same image
3. Sequential refinement - progressively refine segmentation results

Output visualizations are saved to `outputs/` directory.

## Project Structure

```
sam-vision/
├── README.md              
├── requirements.txt     
├── sam.py                 
├── explore_sam3.py        
└── outputs/              
```

## Key Features

- **GPU-optimized**: Uses CUDA for fast inference
- **Open-vocabulary**: Segment any concept described in text
- **Multi-modal prompts**: Combine text and visual prompts
- **Batch processing**: Efficient processing of multiple images

## References

- [SAM3 HuggingFace Model](https://huggingface.co/facebook/sam3)
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
