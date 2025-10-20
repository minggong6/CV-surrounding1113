# sam_segmenter.py

import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import SimpleITK as sitk

# --- Model Loading (cached for performance) ---
@st.cache_resource
def load_sam_model():
    """
    Loads the SAM model and predictor.
    NOTE: You need to download the model checkpoint first.
    e.g., from https://github.com/facebookresearch/segment-anything
    """
    model_type = "vit_b"
    # Update this path to where you've saved the model checkpoint
    checkpoint_path = "./sam_vit_b_01ec64.pth" 
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SamPredictor(sam)
    return predictor

# --- Segmentation Logic ---
def run_sam_segmentation(image_2d, points, labels):
    """
    Runs SAM prediction on a 2D image slice using point prompts.
    
    Args:
        image_2d (np.array): A single 2D slice of the medical image.
        points (np.array): An array of (x, y) coordinates for the prompts.
        labels (np.array): An array of labels (1 for foreground, 0 for background).

    Returns:
        np.array: A 2D boolean mask of the segmented object.
    """
    predictor = load_sam_model()
    
    # Convert grayscale to 3-channel RGB for SAM
    rgb_image = np.stack((image_2d,) * 3, axis=-1).astype(np.uint8)
    
    predictor.set_image(rgb_image)
    
    masks, scores, logits = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=False,
    )
    
    return masks[0]

def create_combined_mask(segmented_masks):
    """
    Combines individual masks into a single multi-label NIfTI mask.
    
    Args:
        segmented_masks (dict): A dictionary like {'artery': [mask1, ...], 'tumor': ...}
        
    Returns:
        np.array: A 3D array with labels 1 (artery), 2 (tumor), 3 (vein).
    """
    if not segmented_masks:
        return None
        
    # Get the shape from the first mask
    ref_mask = next(iter(segmented_masks.values()))[0]
    final_mask = np.zeros_like(ref_mask, dtype=np.uint8)
    
    # Layer the masks according to labels
    if 'artery' in segmented_masks:
        for mask in segmented_masks['artery']:
            final_mask[mask] = 1
            
    if 'vein' in segmented_masks:
        for mask in segmented_masks['vein']:
            final_mask[mask] = 3 # Vein is 3

    if 'tumor' in segmented_masks:
        for mask in segmented_masks['tumor']:
            final_mask[mask] = 2 # Tumor is 2, layered on top

    return final_mask