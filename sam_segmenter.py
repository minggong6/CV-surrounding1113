# sam_segmenter.py (已修正 mask_threshold 错误)

import numpy as np
import torch
import cv2
import os
import sys
import streamlit as st

# --- Add the correct local SAM 2 library path to Python's system path ---
# Copied directly from user's script
sam2_parent_directory = r"D:\CV-surrounding1113(3)\sam2" # Using r"" to handle backslashes

if sam2_parent_directory not in sys.path:
    sys.path.append(sam2_parent_directory)
    # st.info(f"Added SAM 2 library path to sys.path: {sam2_parent_directory}")
# else:
    # st.info(f"SAM 2 library path already in sys.path: {sam2_parent_directory}")

# --- SAM 2 Library Imports (From local path) ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    # st.success("Successfully imported SAM 2 components from local path.")
except ImportError as e:
    st.error(f"ERROR: Failed to import SAM 2 library from '{sam2_parent_directory}'.")
    st.error(f"Please check your path. Details: {e}")
    SAM2_AVAILABLE = False
# --- End Imports ---


@st.cache_resource
def load_sam2_model(model_type="sam2.1_hiera_s", checkpoint_name="sam2.1_hiera_small.pt"):
    """
    Loads the SAM 2 model using the exact path logic from the user's script.
    """
    if not SAM2_AVAILABLE:
        st.error("SAM 2 library not available, cannot load model.")
        return None

    st.info(f"Attempting to load SAM 2 model ({model_type})...")
    try:
        # --- Build the EXACT config file path copied from user's script ---
        # Note the specific subfolder structure: sam2/configs/sam2.1
        model_cfg = os.path.join(
            sam2_parent_directory,
            "sam2",
            "configs",
            "sam2.1",
            f"{model_type}.yaml"
        )
        # --- End copied path logic ---

        st.info(f"Checking for config file at: {model_cfg}")
        if not os.path.exists(model_cfg):
            st.error(f"FATAL: Config file not found at the specified path.")
            st.error("Please ensure the folder structure is exactly D:\\...\\sam2\\sam2\\configs\\sam2.1\\")
            return None
        # st.success("Config file found.") # Reduce messages

        # --- Check for the checkpoint file using relative path from user's script ---
        # Assumes 'checkpoints' is a folder in the *current working directory*
        # where streamlit run app_streamlit.py is executed.
        checkpoint_path = os.path.join("checkpoints", checkpoint_name)
        # --- End copied path logic ---

        st.info(f"Checking for model weights relative to execution directory at: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            st.error(f"FATAL: Model weights file not found!")
            st.error(f"Ensure '{checkpoint_name}' is in a folder named 'checkpoints' located in the directory where you run `streamlit run ...`")
            return None # Stop if relative path fails as per user script logic
        # st.success("Model weights file found.") # Reduce messages

        # --- Load the model ---
        st.info("Loading model into memory...")
        # Call build_sam2 with the constructed config path and the (relative or absolute) checkpoint path
        sam2 = build_sam2(config_file=model_cfg, checkpoint=checkpoint_path)
        sam2.eval()

        if torch.cuda.is_available():
            # st.info("CUDA available. Moving model to GPU.")
            sam2.to("cuda")
        # else:
            # st.warning("CUDA not available. Model running on CPU (slow).")

        predictor = SAM2ImagePredictor(sam2)
        st.success("SAM 2 model loaded successfully!")
        return predictor

    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        import traceback
        traceback.print_exc() # Print detailed error stack trace
        return None

# --- Image Preprocessing ---
def preprocess_for_sam(image_slice):
    if image_slice is None: return None
    if image_slice.ndim != 2:
        image_slice = np.squeeze(image_slice)
        if image_slice.ndim != 2:
             st.error(f"Preprocessing Error: Expected 2D image, got shape {image_slice.shape} after squeeze.")
             return None

    if image_slice.dtype != np.uint8:
        min_val, max_val = np.min(image_slice), np.max(image_slice)
        if max_val > min_val:
            image_slice = ((image_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            image_slice = np.zeros_like(image_slice, dtype=np.uint8)

    rgb_image = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2RGB)
    return rgb_image

# --- Point Prompt Prediction ---
def run_sam2_prediction(predictor, image_slice, points, labels):
    if predictor is None: st.error("SAM predictor not loaded."); return None
    if not points: st.warning("No point prompts provided."); return None

    rgb_image = preprocess_for_sam(image_slice)
    if rgb_image is None: return None

    try:
        predictor.set_image(rgb_image) # 移除了 'image_format' 参数
    except Exception as e:
        st.error(f"SAM predictor.set_image failed: {e}"); return None

    try:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=False,
        )
        # --- !!! 修正点 1 !!! ---
        # 使用 0.0 作为标准阈值，而不是访问不存在的属性
        return masks[0] > 0.0 
    except Exception as e:
        st.error(f"SAM predictor.predict failed: {e}"); return None


# --- Mask Prompt Refinement ---
def run_sam2_refinement_with_mask(predictor, image_slice_numpy, mask_prompt_slice_numpy):
    if predictor is None: st.error("SAM predictor not loaded."); return None
    if image_slice_numpy is None or mask_prompt_slice_numpy is None:
        st.error("Input image or mask prompt is None."); return None

    rgb_image = preprocess_for_sam(image_slice_numpy)
    if rgb_image is None: return None

    try:
        predictor.set_image(rgb_image) # 移除了 'image_format' 参数
    except Exception as e:
        st.error(f"SAM predictor.set_image (with mask) failed: {e}"); return None

    try:
        mask_prompt_torch = torch.as_tensor(
            mask_prompt_slice_numpy.astype(float),
            device=predictor.model.device
        ).unsqueeze(0)
    except Exception as e:
        st.error(f"Failed to convert mask prompt to tensor: {e}"); return None

    try:
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            mask_input=mask_prompt_torch,
            multimask_output=False
        )
        # --- !!! 修正点 2 !!! ---
        # 使用 0.0 作为标准阈值，而不是访问不存在的属性
        return masks[0] > 0.0 
    except AttributeError as ae:
         st.error(f"Error calling predictor: {ae}. Check SAM2ImagePredictor arguments.")
         return None
    except Exception as e:
        st.error(f"SAM predictor.predict (with mask) failed: {e}"); return None