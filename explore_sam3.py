"""
Simple SAM3 Exploration App
Demonstrates the capabilities of Meta's SAM3 model for image and video segmentation
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import urllib.request
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

matplotlib.use('Agg')  # Non-interactive backend

class SAM3Explorer:
    def __init__(self):
        """Initialize SAM3 model and processor (GPU required)"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required to run SAM3. Please ensure NVIDIA GPU support is available.")
        
        self.device = "cuda"
        print(f"Loading SAM3 model on {self.device}...")
        
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        print("✓ Model loaded successfully!")
        
    def explore_text_prompt(self, image_path: str, text_prompt: str):
        """
        Explore text-based segmentation
        Segment objects by describing them with text
        """
        print(f"\n{'='*60}")
        print(f"EXPLORATION 1: Text-Based Segmentation")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Text prompt: '{text_prompt}'")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"Image size: {image.size}")
        
        # Set image and text prompt
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        print(f"✓ Found {len(masks)} object(s)")
        if len(masks) > 0:
            print(f"  Confidence scores: {scores}")
            print(f"  Bounding boxes shape: {boxes.shape if hasattr(boxes, 'shape') else len(boxes)}")
        
        return image, {"masks": masks, "boxes": boxes, "scores": scores}
    
    def explore_multiple_prompts(self, image_path: str, text_prompts: list):
        """
        Explore multiple text prompts on same image
        Segment different concepts sequentially
        """
        print(f"\n{'='*60}")
        print(f"EXPLORATION 2: Multiple Text Prompts")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Text prompts: {text_prompts}")
        
        image = Image.open(image_path).convert("RGB")
        
        results_list = []
        for prompt in text_prompts:
            inference_state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            results_list.append({
                "prompt": prompt,
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
                "count": len(masks)
            })
            print(f"  '{prompt}': {len(masks)} object(s) found")
        
        return image, results_list
    
    def explore_sequential_refinement(self, image_path: str, initial_prompt: str, refine_prompt: str):
        """
        Explore sequential refinement with different prompts
        First segment with one concept, then another
        """
        print(f"\n{'='*60}")
        print(f"EXPLORATION 3: Sequential Refinement")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"First prompt: '{initial_prompt}'")
        print(f"Refine prompt: '{refine_prompt}'")
        
        image = Image.open(image_path).convert("RGB")
        
        # First pass
        print(f"\n  Pass 1: Segmenting '{initial_prompt}'...")
        inference_state = self.processor.set_image(image)
        output1 = self.processor.set_text_prompt(state=inference_state, prompt=initial_prompt)
        
        masks1 = output1["masks"]
        print(f"    Found {len(masks1)} object(s)")
        
        # Second pass with different concept
        print(f"\n  Pass 2: Segmenting '{refine_prompt}'...")
        inference_state = self.processor.set_image(image)
        output2 = self.processor.set_text_prompt(state=inference_state, prompt=refine_prompt)
        
        masks2 = output2["masks"]
        print(f"    Found {len(masks2)} object(s)")
        
        return image, {
            "pass1": {"prompt": initial_prompt, "masks": masks1, "boxes": output1["boxes"], "scores": output1["scores"]},
            "pass2": {"prompt": refine_prompt, "masks": masks2, "boxes": output2["boxes"], "scores": output2["scores"]}
        }
    
    def visualize_results(self, image, results, output_path: str):
        """
        Visualize segmentation results with colored overlays
        """
        # Handle different result formats
        if isinstance(results, list):
            # Multiple prompts case - visualize first one
            masks = results[0]["masks"]
        elif isinstance(results, dict) and "pass1" in results:
            # Sequential refinement - combine masks from both passes
            masks1 = results["pass1"]["masks"]
            masks2 = results["pass2"]["masks"]
            masks = np.concatenate([masks1, masks2], axis=0) if len(masks2) > 0 else masks1
        else:
            masks = results["masks"]
        
        if len(masks) == 0:
            print(f"  No masks to visualize")
            return
        
        image = image.convert("RGBA")
        masks = np.asarray(masks)
        
        # Ensure masks are numpy arrays with correct shape
        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=0)
        
        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(n_masks, 1))
        colors = [
            tuple(int(c * 255) for c in cmap(i % n_masks)[:3])
            for i in range(n_masks)
        ]
        
        for mask, color in zip(masks, colors):
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
            mask_img = Image.fromarray(mask_uint8)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask_img.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        
        image.save(output_path)
        print(f"  Visualization saved: {output_path}")
    
    def download_sample_image(self, output_path: str = "sample_image.jpg"):
        """
        Download a sample image for testing
        """
        url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        print(f"Downloading sample image...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Sample image saved: {output_path}")
            return output_path
        except Exception as e:
            print(f"✗ Failed to download: {e}")
            return None


def main():
    """Main exploration routine"""
    print("\n" + "="*60)
    print("SAM3 EXPLORATION APP")
    print("="*60)
    print("\nThis app explores the capabilities of Meta's SAM3 model")
    print("SAM3: Unified foundation model for promptable segmentation\n")
    
    # Initialize explorer
    try:
        explorer = SAM3Explorer()
    except RuntimeError as e:
        print(f"✗ {e}")
        return
    except Exception as e:
        print(f"✗ Failed to initialize SAM3: {e}")
        return
    
    # Download sample image for testing
    sample_image = explorer.download_sample_image("sample_image.jpg")
    if not sample_image:
        print("✗ Cannot proceed without sample image")
        return
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # EXPLORATION 1: Text-based segmentation
    try:
        image, results = explorer.explore_text_prompt(sample_image, "cat")
        explorer.visualize_results(image, results, "outputs/exploration_1_cat.png")
    except Exception as e:
        print(f"✗ Exploration 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # EXPLORATION 2: Multiple prompts
    try:
        image, results = explorer.explore_multiple_prompts(sample_image, ["cat", "whiskers", "ear"])
        explorer.visualize_results(image, results, "outputs/exploration_2_multiple_prompts.png")
    except Exception as e:
        print(f"✗ Exploration 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # EXPLORATION 3: Sequential refinement
    try:
        image, results = explorer.explore_sequential_refinement(sample_image, "animal", "face")
        explorer.visualize_results(image, results, "outputs/exploration_3_refinement.png")
    except Exception as e:
        print(f"✗ Exploration 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✓ SAM3 Exploration Complete!")
    print(f"{'='*60}")
    print("\nKey SAM3 Capabilities Demonstrated:")
    print("  1. Text-based segmentation (open-vocabulary)")
    print("  2. Multiple concept segmentation")
    print("  3. Sequential prompt refinement")
    print("\nCheck 'outputs/' directory for visualizations")
    print("\nNext Steps:")
    print("  - Try with your own images")
    print("  - Explore video segmentation capabilities")
    print("  - Experiment with different object descriptions")
    print("  - Test concept segmentation on complex scenes\n")


if __name__ == "__main__":
    main()
