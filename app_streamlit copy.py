import os
import tempfile
import shutil
import traceback

import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# reuse project modules
import toolkit_main as tkm
import toolkit_3D as tk3
from contact3D import calculate_3D_contact
from contact2D import calculate_2D_contact

# Make tk3.get_nii tolerant to a 'rotate' keyword if the underlying function doesn't accept it.
# This avoids TypeError in places that call tk3.get_nii(..., rotate=True)pyt
try:
    _orig_get_nii = tk3.get_nii

    def _get_nii_compat(path, *args, **kwargs):
        if 'rotate' in kwargs:
            try:
                return _orig_get_nii(path, *args, **kwargs)
            except TypeError:
                # underlying implementation doesn't accept 'rotate'
                kwargs.pop('rotate')
                return _orig_get_nii(path, *args, **kwargs)
        else:
            return _orig_get_nii(path, *args, **kwargs)

    tk3.get_nii = _get_nii_compat
except Exception:
    # If monkeypatching fails for some reason, continue — safe_get_nii also handles compatibility.
    pass

# skeleton analysis may fail if kimimaro/cloud-volume incompatible
try:
    from SkeletonAnalysis import skeleton_analysis
    SKELETON_AVAILABLE = True
except Exception:
    SKELETON_AVAILABLE = False


st.set_page_config(page_title="Pancreas Resectability Demo", layout="wide")
st.title("Pancreas Resectability — Quick Demo")

st.markdown("Upload a segmented NIfTI (.nii or .nii.gz) where labels are: 1=artery, 2=tumor, 3=vein.")

uploaded = st.file_uploader("Upload segmentation NIfTI", type=["nii", "nii.gz"])

with st.sidebar:
    st.header("Parameters")
    contour_thickness = st.slider("Contour thickness", 0.5, 5.0, 1.5)
    contact_range = st.slider("Contact range (voxels)", 0, 10, 2)
    axis = st.selectbox("Axis for slice view", ["z", "x"], index=0)
    do_2d = st.checkbox("Run 2D contact analysis", value=True)
    do_3d = st.checkbox("Run 3D contact analysis", value=True)
    do_skeleton = st.checkbox("Run skeleton analysis (may fail if kimimaro not installed)", value=False)
    if do_skeleton and not SKELETON_AVAILABLE:
        st.warning("SkeletonAnalysis import failed in this environment. Skeleton option will be skipped.")
        do_skeleton = False


def save_uploaded(tmpdir, uploaded_file):
    dest = os.path.join(tmpdir, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def load_normalized_nii(path):
    """
    Load a NIfTI and return a 3D numpy array with shape (H, W, Z) (label volume).
    Handles:
      - channel-first one-hot e.g. (C, H, W) or (C, H, W, T) -> move channel to last, argmax
      - channel-last one-hot e.g. (H, W, C) -> argmax
      - 2D arrays -> expand to (H, W, 1)
      - already (H, W, Z) -> pass-through
    """
    img = nib.load(path)
    data = np.asarray(img.get_fdata())
    # 4D where first dim is small -> treat as channel-first
    if data.ndim == 4 and data.shape[0] <= 8:
        data = np.moveaxis(data, 0, -1)
    # 3D but one axis is small (<=8) and others large -> likely channel-first
    if data.ndim == 3:
        small_axes = [i for i, s in enumerate(data.shape) if s <= 8]
        if small_axes and max(data.shape) > 20:
            ch = small_axes[0]
            if ch != (data.ndim - 1):
                data = np.moveaxis(data, ch, -1)
    # If last axis looks like channels (<=8), convert one-hot/multi-channel to labels
    if data.ndim == 3 and data.shape[-1] <= 8:
        try:
            data = np.argmax(data, axis=-1)
        except Exception:
            data = np.squeeze(data)
    # If 2D -> expand to single-slice 3D
    if data.ndim == 2:
        data = data[..., np.newaxis]
    # final check/reshape
    data = np.asarray(data)
    if data.ndim == 3:
        return data
    data = np.squeeze(data)
    if data.ndim == 2:
        return data[..., np.newaxis]
    raise ValueError(f"Unsupported nifti shape after normalization: {data.shape}")


def display_slice_from_nii(nii_path, axis='z', slice_index=None):
    img = load_normalized_nii(nii_path)
    # rotate/assume orientation similar to toolkit_3D.get_nii
    if axis == 'z':
        depth = img.shape[2]
        if slice_index is None:
            slice_index = depth // 2
        slice_img = img[:, :, slice_index]
    else:
        depth = img.shape[0]
        if slice_index is None:
            slice_index = depth // 2
        slice_img = img[slice_index, :, :]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(slice_img.T, cmap='gray', origin='lower')
    ax.set_title(f"Slice {slice_index} (axis={axis})")
    ax.axis('off')
    return fig


def make_overlay_fig(origin_nii, contact_nii, axis='z', slice_index=None):
    oimg = load_normalized_nii(origin_nii)
    cimg = load_normalized_nii(contact_nii)
    if axis == 'z':
        if slice_index is None:
            slice_index = oimg.shape[2] // 2
        o = oimg[:, :, slice_index]
        c = cimg[:, :, slice_index]
    else:
        if slice_index is None:
            slice_index = oimg.shape[0] // 2
        o = oimg[slice_index, :, :]
        c = cimg[slice_index, :, :]

    # origin: show tumor in red, artery/vein contours and contacts with colors
    base = np.zeros((o.shape[0], o.shape[1], 3), dtype=np.float32)
    # background: remain dark
    # tumor (value 2 in segmentation) -> show as red channel
    tumor_mask = (o == 2)
    base[..., 0] = tumor_mask.astype(float) * 0.8
    # artery contour (2) and contact labels: contact_ set encodes different codes; use simple masks
    artery_contour = (c == 2) | (c == 4)
    artery_contact = (c == 3) | (c == 5)
    vein_contour = (c == 2)  # fallback (depending on how saved)
    vein_contact = (c == 3)
    base[..., 1] = artery_contour.astype(float) * 0.6 + vein_contact.astype(float) * 0.4
    base[..., 2] = artery_contact.astype(float) * 0.8 + vein_contour.astype(float) * 0.3

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(base.T, origin='lower')
    ax.set_title(f"Overlay slice {slice_index} (axis={axis})")
    ax.axis('off')
    return fig


def safe_get_nii(path, rotate=True):
    """Call tk3.get_nii in a backward-compatible way: try keyword 'rotate' first,
    fall back to calling without keyword if function signature doesn't accept it.
    """
    try:
        return tk3.get_nii(path, rotate=rotate)
    except TypeError:
        # fallback to positional call if rotate kw not supported
        return tk3.get_nii(path)


if uploaded is not None:
    tmpdir = tempfile.mkdtemp(prefix="pancreas_demo_")
    try:
        saved = save_uploaded(tmpdir, uploaded)
        st.success(f"Saved uploaded file to {saved}")

        file_dict = {
            "img_id": os.path.splitext(os.path.basename(saved))[0],
            "img_path": saved,
            "img_contact_path": None
        }

        contact_dir = tmpdir

        st.info("Generating contact image (this may take a few seconds)...")
        with st.spinner("Generating contact..."):
            try:
                tkm.generate_contour_nii_3D(file_dict, contact_dir, prefix="contact_", contour_thickness=contour_thickness,
                                            contact_range=contact_range, axis=axis)
            except Exception as e:
                st.error("Failed to generate contact: " + str(e))
                st.text(traceback.format_exc())

        st.write("Contact NIfTI:", file_dict.get("img_contact_path"))

        results = {"2D": {}, "3D": {}, "skeleton": {}}

        if do_2d:
            st.info("Running 2D contact analysis...")
            with st.spinner("2D analyzing..."):
                for target in ["vein", "artery"]:
                    try:
                        slice_list, max_ratio, max_slice = tkm.calculate_2D_contact(file_dict, target,
                                                                                    contact_dir=contact_dir,
                                                                                    size_threshold=30, axis=axis)
                        results["2D"][target] = {"slices": slice_list, "max_ratio": float(max_ratio), "max_slice": int(max_slice)}
                    except Exception as e:
                        results["2D"][target] = {"error": str(e)}
                        st.error(f"2D analysis failed for {target}: {e}")

        if do_3d:
            st.info("Running 3D contact analysis...")
            with st.spinner("3D analyzing..."):
                for target in ["vein", "artery"]:
                    try:
                        res3 = calculate_3D_contact(file_dict, target)
                        results["3D"][target] = res3
                    except Exception as e:
                        results["3D"][target] = {"error": str(e)}
                        st.error(f"3D analysis failed for {target}: {e}")

        if do_skeleton:
            st.info("Running skeleton analysis...")
            with st.spinner("Skeletonizing..."):
                for target in ["vein", "artery"]:
                    try:
                        skeleton_res = skeleton_analysis(file_dict, target, print_info=False)
                        results["skeleton"][target] = skeleton_res
                    except Exception as e:
                        results["skeleton"][target] = {"error": str(e)}
                        st.error(f"Skeleton analysis failed for {target}: {e}")

        # tumor info
        try:
            # Prefer toolkit get_nii result if available (some toolkit returns dict with masks)
            origin_struct = None
            try:
                origin_struct = safe_get_nii(file_dict["img_path"], rotate=True)
            except Exception:
                origin_struct = None
            if isinstance(origin_struct, dict) and "tumor" in origin_struct:
                tumor_mask = origin_struct["tumor"]
                tumor_volume = int(np.sum(np.asarray(tumor_mask) > 0))
            else:
                # fallback: load normalized label volume and compute tumor==2
                lbl = load_normalized_nii(file_dict["img_path"])
                tumor_volume = int(np.sum(lbl == 2))
        except Exception:
            tumor_volume = None

        # simple resectability scoring rule (example)
        score = 0.5
        # combine 3D artery ratio if available
        if "artery" in results["3D"] and isinstance(results["3D"]["artery"], list) and len(results["3D"]["artery"]) > 0:
            c3a = results["3D"]["artery"][0].get("contact_ratio", 0)
            score -= 0.3 * c3a
        if "vein" in results["3D"] and isinstance(results["3D"]["vein"], list) and len(results["3D"]["vein"]) > 0:
            c3v = results["3D"]["vein"][0].get("contact_ratio", 0)
            score -= 0.15 * c3v
        if tumor_volume is not None:
            # normalize roughly by an arbitrary factor
            score -= 0.05 * np.log1p(tumor_volume)
        score = float(max(0.0, min(1.0, score)))

        if score > 0.7:
            label = "likely resectable"
        elif score > 0.4:
            label = "borderline"
        else:
            label = "likely unresectable"

        st.header("Results summary")
        st.write("Resectability score:", score, "(", label, ")")
        st.json(results)

        # visualization: show overlay slice
        st.header("Visualization")
        if file_dict.get("img_contact_path") and os.path.exists(file_dict["img_contact_path"]):
            oimg = load_normalized_nii(file_dict["img_path"])  # 兼容多种输入形状，返回 (H,W,Z)
            if axis == 'z':
                # 当axis='z'时，切片轴为第2轴（索引2）
                max_slice = oimg.shape[2] - 1
            else:
                # 当axis='x'时，切片轴为第0轴（索引0）
                max_slice = oimg.shape[0] - 1
     # 确保滑块最大索引不超过max_slice
                slice_index = st.slider(
                 "Select slice index",
                 min_value=0,
                 max_value=max_slice,
                 value=min(int(oimg.shape[2]//2 if axis=='z' else oimg.shape[0]//2), max_slice)  # 初始值限制在有效范围
             )
     # 调用函数生成图像
                fig = make_overlay_fig(file_dict["img_path"], file_dict["img_contact_path"], axis=axis, slice_index=slice_index)
                st.pyplot(fig)
        else:
            st.info("No contact NIfTI produced to visualize.")

    finally:
        # keep tmp for debugging; if you want, remove the directory
        if st.checkbox("Clean temp files now", value=False):
            try:
                shutil.rmtree(tmpdir)
                st.success("Temporary files removed")
            except Exception as e:
                st.error("Failed to remove temp dir: " + str(e))

else:
    st.info("Please upload a segmented NIfTI to start the analysis.")
