# sam_segmenter.py (with validation code)

import numpy as np
import torch
import cv2
import os
import sys

# --- Mock Streamlit for direct execution ---
# This allows us to run the script without Streamlit, by creating dummy functions.
class StreamlitMock:
    def info(self, msg): print(f"INFO: {msg}")
    def success(self, msg): print(f"SUCCESS: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def cache_resource(self, func): return func

st = StreamlitMock()
# --- End Mock ---


# --- Add the correct local SAM 2 library path to Python's system path ---
# 修正了文件夹名，从 (3 改为 (3)
sam2_parent_directory = "D:\\CV-surrounding1113(3)\\sam2"

if sam2_parent_directory not in sys.path:
    sys.path.append(sam2_parent_directory)

# --- SAM 2 Library Imports ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("SUCCESS: SAM 2 library components imported successfully.")
except ImportError as e:
    print(f"ERROR: Failed to import SAM 2 library from '{sam2_parent_directory}'.")
    print(f"Please check your path. Details: {e}")
    SAM2_AVAILABLE = False


def load_sam2_model(model_type="sam2.1_hiera_s", checkpoint_name="sam2.1_hiera_small.pt"):
    """
    Loads the SAM 2 model using the corrected config path.
    """
    if not SAM2_AVAILABLE:
        return None

    st.info(f"Attempting to load SAM 2 model ({model_type})...")
    try:
        # --- Build the EXACT config file path with the corrected directory name ---
        model_cfg = os.path.join(
            sam2_parent_directory, 
            "sam2", 
            "configs", 
            "sam2.1", 
            f"{model_type}.yaml"
        )
        
        st.info(f"Checking for config file at: {model_cfg}")
        if not os.path.exists(model_cfg):
            st.error(f"FATAL: Config file not found at the specified path.")
            st.error("Please double-check your folder structure and file names.")
            return None
        st.success("Config file found.")

        # --- Check for the checkpoint file ---
        checkpoint_path = f"checkpoints/{checkpoint_name}"
        st.info(f"Checking for model weights at: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            st.error(f"FATAL: Model weights file not found! Ensure '{checkpoint_name}' is in the 'checkpoints' folder.")
            return None
        st.success("Model weights file found.")

        # --- Load the model ---
        st.info("Loading model into memory... (This may take a moment)")
        sam2 = build_sam2(model_cfg, checkpoint_path)
        sam2.eval()
        
        if torch.cuda.is_available():
            st.info("CUDA is available. Moving model to GPU.")
            sam2.to("cuda")
        else:
            st.warning("CUDA not available. Model will run on CPU, which may be slow.")
            
        predictor = SAM2ImagePredictor(sam2)
        st.success("SAM 2 model loaded successfully!")
        return predictor
        
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        import traceback
        traceback.print_exc() # Print detailed error stack trace
        return None

# The rest of your functions remain the same...
def preprocess_for_sam(image_slice):
    if image_slice.ndim != 2:
        st.error(f"Image slice should be 2D, but got {image_slice.ndim} dimensions.")
        return None
    img_normalized = cv2.normalize(image_slice, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rgb_image = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    return rgb_image

def run_sam2_prediction(predictor, image_slice, points, labels):
    if predictor is None or not points:
        return None
    rgb_image = preprocess_for_sam(image_slice)
    if rgb_image is None: return None
    predictor.set_image(rgb_image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=False,
    )
    return masks[0]


# ==============================================================================
# --- VALIDATION SCRIPT ---
# This part only runs when you execute `python sam_segmenter.py` directly.
# ==============================================================================
if __name__ == '__main__':
    print("=============================================")
    print("--- Running SAM 2 Loader Validation Script ---")
    print("=============================================")
    
    # We call the function directly to test it.
    model_predictor = load_sam2_model()
    
    print("\n--- Validation Summary ---")
    if model_predictor:
        print("✅  Validation PASSED: The SAM 2 model was loaded successfully.")
    else:
        print("❌  Validation FAILED: The SAM 2 model could not be loaded. Please review the ERROR messages above.")
    
    print("=============================================")