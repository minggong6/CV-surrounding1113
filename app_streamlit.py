# app_streamlit.py (Corrected list comprehension to a for loop)

import os
import tempfile
import shutil
import time
import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# --- Font Configuration ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Project Modules ---
import toolkit_main as tkm
import toolkit_3D as tk3
from contact3D import calculate_3D_contact
from contact2D import calculate_2D_contact
import sam_segmenter

try:
    from SkeletonAnalysis import skeleton_analysis, get_skeleton_img
    SKELETON_AVAILABLE = True
except ImportError:
    st.sidebar.warning("无法导入 SkeletonAnalysis。骨架分析功能将不可用。")
    SKELETON_AVAILABLE = False

# --- Page Config ---
st.set_page_config(page_title="胰腺癌可切除性分析", layout="wide")
st.title("胰腺癌可切除性分析 Demo")

# ==============================================================================
# --- Cached Functions for Performance ---
# ==============================================================================

@st.cache_data(show_spinner=False)
def perform_full_analysis(_uploaded_file_bytes, _file_name, _contour_thickness, _contact_range, _axis, _do_2d, _do_3d, _do_skeleton):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, _file_name)
        with open(file_path, "wb") as f: f.write(_uploaded_file_bytes)

        file_dict = { "img_id": os.path.splitext(_file_name)[0], "img_path": file_path, "img_contact_path": None }

        total_steps = 1 + (1 if _do_2d else 0) + (1 if _do_3d else 0) + (1 if _do_skeleton else 0)
        progress_bar = st.progress(0, text="正在初始化分析...")
        
        def update_progress(step, name):
            progress = step / total_steps
            progress_bar.progress(progress, text=f"步骤 {step}/{total_steps}: {name}")

        update_progress(1, "正在生成接触面图像...")
        tkm.generate_contour_nii_3D(file_dict, tmpdir, prefix="contact_", contour_thickness=_contour_thickness, contact_range=_contact_range, axis=_axis)
        
        results = {"2D": {}, "3D": {}, "skeleton": {}}
        step_count = 1

        if _do_2d:
            step_count += 1; update_progress(step_count, "正在执行 2D 接触分析...")
            for target in ["vein", "artery"]:
                try:
                    res2d, max_r, max_s = tkm.calculate_2D_contact(file_dict, target, contact_dir=tmpdir, size_threshold=30, axis=_axis)
                    results["2D"][target] = {"slices": res2d, "max_ratio": float(max_r), "max_slice": int(max_s)}
                except Exception as e: results["2D"][target] = {"error": str(e)}

        if _do_3d:
            step_count += 1; update_progress(step_count, "正在执行 3D 接触分析...")
            for target in ["vein", "artery"]:
                try: results["3D"][target] = calculate_3D_contact(file_dict, target)
                except Exception as e: results["3D"][target] = {"error": str(e)}
        
        artery_skeleton, vein_skeleton = None, None
        img_nii_dict = tk3.get_nii(file_dict["img_path"], axis=_axis)
        
        if _do_skeleton and SKELETON_AVAILABLE:
            step_count += 1; update_progress(step_count, "正在执行骨架分析 (此步耗时最长)...")
            for target in ["vein", "artery"]:
                try:
                    results["skeleton"][target] = skeleton_analysis(file_dict, target, print_info=False)
                    if target == "artery": artery_skeleton = get_skeleton_img(img_nii_dict[target], expand=1)
                    else: vein_skeleton = get_skeleton_img(img_nii_dict[target], expand=1)
                except Exception as e: results["skeleton"][target] = {"error": str(e)}
        
        original_img_data = img_nii_dict["origin"]
        contact_img_data = tk3.get_any_nii(file_dict["img_contact_path"], axis=_axis)['img']

        progress_bar.progress(1.0, text="分析完成！")
        return results, original_img_data, contact_img_data, artery_skeleton, vein_skeleton

# ==============================================================================
# --- New Resectability Advisor Module ---
# ==============================================================================
def display_resectability_recommendation(results):
    st.header("可切除性评估建议 (Resectability Assessment)")
    
    artery_contact_ratio, vein_contact_ratio = 0.0, 0.0
    if "3D" in results and results["3D"]:
        if "artery" in results["3D"] and isinstance(results["3D"]["artery"], list) and results["3D"]["artery"]:
            artery_contact_ratio = max([seg.get("contact_ratio", 0) for seg in results["3D"]["artery"]])
        if "vein" in results["3D"] and isinstance(results["3D"]["vein"], list) and results["3D"]["vein"]:
            vein_contact_ratio = max([seg.get("contact_ratio", 0) for seg in results["3D"]["vein"]])

    UNRESECTABLE_ARTERY_THRESHOLD, BORDERLINE_VEIN_THRESHOLD = 0.5, 0.5
    recommendation, reasons = "🟢 **可切除 (Resectable)**", []

    if artery_contact_ratio > UNRESECTABLE_ARTERY_THRESHOLD:
        recommendation = "🔴 **不可切除 (Unresectable)**"
        reasons.append(f"**主要动脉包裹**: 肿瘤与动脉的最大接触比例为 **{artery_contact_ratio:.2%}**，超过了 180° 包裹的阈值 ({UNRESECTABLE_ARTERY_THRESHOLD:.0%})，这是不可切除的关键指标。")
    elif vein_contact_ratio > BORDERLINE_VEIN_THRESHOLD:
        recommendation = "🟡 **交界可切除 (Borderline Resectable)**"
        reasons.append(f"**主要静脉包裹**: 肿瘤与静脉的最大接触比例为 **{vein_contact_ratio:.2%}**，超过了 180° 包裹的阈值 ({BORDERLINE_VEIN_THRESHOLD:.0%})，需要进一步评估是否可以通过血管重建进行切除。")
    elif artery_contact_ratio > 0:
        recommendation = "🟡 **交界可切除 (Borderline Resectable)**"
        reasons.append(f"**动脉邻接**: 肿瘤与动脉存在接触（最大比例 **{artery_contact_ratio:.2%}**），但未达到完全包裹的程度，属于交界可切除范畴。")
    else:
        reasons.append("肿瘤与主要动脉无接触，且与主要静脉的接触未达到完全包裹的程度，具备良好的手术切除条件。")

    st.markdown(f"### 评估结果: {recommendation}")
    with st.container():
        st.markdown("**评估依据:**")
        # --- FIX: Replaced list comprehension with a standard for loop ---
        for r in reasons:
            st.markdown(f"- {r}")
        # --- END FIX ---
        st.markdown(f"**关键参数:**")
        st.markdown(f"  - **动脉最大接触比例**: `{artery_contact_ratio:.2%}`")
        st.markdown(f"  - **静脉最大接触比例**: `{vein_contact_ratio:.2%}`")
        st.caption("注：该建议基于 NCCN 指南和相关论文的关键思想，将 3D 接触比例 > 50% 作为血管被包裹 > 180° 的近似标准。此结果仅供参考。")

# ==============================================================================
# --- Visualization Functions (Now operate on in-memory data) ---
# ==============================================================================
def make_overlay_fig(original_img_data, contact_img_data, axis='z', slice_index=None):
    oimg, cimg = original_img_data, contact_img_data
    if axis == 'z':
        if slice_index is None: slice_index = oimg.shape[2] // 2
        o, c = oimg[:, :, slice_index], cimg[:, :, slice_index]
    else:
        if slice_index is None: slice_index = oimg.shape[0] // 2
        o, c = oimg[slice_index, :, :], cimg[slice_index, :, :]
    
    base = np.zeros((o.shape[1], o.shape[0], 3), dtype=np.float32)
    base[..., 0] = (o == 2).T.astype(float) * 0.8
    base[..., 1] = (((c == 4) | (c == 2)).T.astype(float) * 0.6) + ((c == 3).T.astype(float) * 0.4)
    base[..., 2] = (((c == 5) | (c == 3)).T.astype(float) * 0.8) + ((c == 2).T.astype(float) * 0.3)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(base, origin='lower')
    ax.set_title(f"叠加影像 (Overlay) - 切片 {slice_index} (轴={axis})"); ax.axis('off')
    return fig

def make_skeleton_fig(skeleton_img_3d, title, axis='z', slice_index=None):
    if axis == 'z':
        if slice_index is None: slice_index = skeleton_img_3d.shape[2] // 2
        slice_data = skeleton_img_3d[:, :, slice_index]
    else:
        if slice_index is None: slice_index = skeleton_img_3d.shape[0] // 2
        slice_data = skeleton_img_3d[slice_index, :, :]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(slice_data.T, cmap='hot', origin='lower')
    ax.set_title(f"{title} - 切片 {slice_index} (轴={axis})"); ax.axis('off')
    return fig

# ==============================================================================
# --- Sidebar UI & Main App Logic ---
# ==============================================================================
with st.sidebar:
    st.header("1. 工作模式"); mode = st.radio("选择分割方式", ('上传已分割文件', '使用 SAM 2 分割'))
    st.markdown("---"); st.header("2. 分析参数")
    contour_thickness = st.slider("轮廓厚度", 0.5, 5.0, 1.5)
    contact_range = st.slider("接触范围", 0, 10, 2)
    axis = st.selectbox("2D 观察轴", ["z", "x"], index=0)
    st.markdown("---"); st.header("3. 分析模块")
    do_2d = st.checkbox("2D 接触分析", value=True)
    do_3d = st.checkbox("3D 接触分析", value=True)
    do_skeleton = st.checkbox("骨架分析", value=True)
    if do_skeleton and not SKELETON_AVAILABLE: st.warning("骨架分析模块不可用。"); do_skeleton = False

if mode == '上传已分割文件':
    st.header("上传已分割的 NIfTI 文件")
    st.markdown("标签定义: 1=动脉, 2=肿瘤, 3=静脉。")
    uploaded_file = st.file_uploader("上传分割文件", type=["nii", "nii.gz"])

    if uploaded_file:
        results, original_img, contact_img, artery_skeleton, vein_skeleton = perform_full_analysis(
            uploaded_file.getvalue(), uploaded_file.name, contour_thickness, contact_range, axis, do_2d, do_3d, do_skeleton
        )
        display_resectability_recommendation(results)
        with st.expander("点击查看详细分析数据"): st.json(results)
        
        st.header("可视化")
        if original_img is not None and contact_img is not None:
            max_slice = original_img.shape[2] - 1 if axis == 'z' else original_img.shape[0] - 1
            default_slice = original_img.shape[2] // 2 if axis == 'z' else original_img.shape[0] // 2
            slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice))
            st.pyplot(make_overlay_fig(original_img, contact_img, axis=axis, slice_index=slice_index))
            
            if do_skeleton and SKELETON_AVAILABLE:
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("显示动脉骨架") and artery_skeleton is not None: st.pyplot(make_skeleton_fig(artery_skeleton, "动脉骨架", axis, slice_index))
                with col2:
                    if st.checkbox("显示静脉骨架") and vein_skeleton is not None: st.pyplot(make_skeleton_fig(vein_skeleton, "静脉骨架", axis, slice_index))
        else:
            st.info("未能生成接触面影像，无法进行可视化。")


elif mode == '使用 SAM 2 进行实时分割':
    st.header("使用 SAM 2 进行实时分割")
    st.markdown("请上传一个**原始的、未分割的**医学影像文件 (例如 CT 平扫期)。")
    
    raw_file = st.file_uploader("上传原始影像文件", type=["nii", "nii.gz"])

    if 'masks' not in st.session_state: st.session_state.masks = {'artery': [], 'tumor': [], 'vein': []}
    if 'points' not in st.session_state: st.session_state.points = []
    if 'labels' not in st.session_state: st.session_state.labels = []

    if raw_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = save_uploaded_file(raw_file, tmpdir)
            raw_img_data = nib.load(raw_path).get_fdata()

            col1, col2 = st.columns([2, 1])
            with col1:
                slice_idx = st.slider("选择要标注的切片", 0, raw_img_data.shape[2] - 1, raw_img_data.shape[2] // 2)
                current_slice = raw_img_data[:, :, slice_idx]
                fig, ax = plt.subplots(); ax.imshow(current_slice.T, cmap='gray', origin='lower'); ax.set_title(f"当前切片: {slice_idx}"); ax.set_axis_off()
                for point, label in zip(st.session_state.points, st.session_state.labels):
                    ax.scatter(point[0], point[1], color='green' if label == 1 else 'red', marker='*', s=100)
                st.pyplot(fig)
            with col2:
                st.subheader("标注工具")
                structure_to_label = st.radio("选择要标注的结构", ('肿瘤', '动脉', '静脉'))
                label_map = {'肿瘤': 'tumor', '动脉': 'artery', '静脉': 'vein'}
                x_coord = st.number_input("输入 X 坐标", 0, current_slice.shape[0]-1, step=1)
                y_coord = st.number_input("输入 Y 坐标", 0, current_slice.shape[1]-1, step=1)
                if st.button("添加前景点"): st.session_state.points.append((x_coord, y_coord)); st.session_state.labels.append(1); st.rerun()
                if st.button("运行 SAM 分割"):
                    predictor = sam_segmenter.load_sam2_model()
                    if predictor and st.session_state.points:
                        with st.spinner("SAM 2 正在分割..."):
                            mask = sam_segmenter.run_sam2_prediction(predictor, current_slice, st.session_state.points, st.session_state.labels)
                            target_key = label_map[structure_to_label]
                            full_mask = np.zeros_like(raw_img_data, dtype=bool); full_mask[:, :, slice_idx] = mask
                            st.session_state.masks[target_key].append(full_mask)
                            st.success(f"已为“{structure_to_label}”添加掩码。")
                            st.session_state.points = []; st.session_state.labels = []; st.rerun()
                    else: st.warning("请至少添加一个前景点。")
                if st.button("清除当前点"): st.session_state.points = []; st.session_state.labels = []; st.rerun()
            
            st.markdown("---")
            st.subheader("完成与分析")
            st.write(f"已分割掩码: 肿瘤: {len(st.session_state.masks['tumor'])}, 动脉: {len(st.session_state.masks['artery'])}, 静脉: {len(st.session_state.masks['vein'])}")

            if st.button("完成所有分割，开始分析"):
                if not any(st.session_state.masks.values()):
                    st.error("请至少完成一个结构的分割。")
                else:
                    with st.spinner("正在合并掩码并分析..."):
                        final_3d_mask = np.zeros_like(raw_img_data, dtype=np.uint8)
                        for m in st.session_state.masks['artery']: final_3d_mask[m] = 1
                        for m in st.session_state.masks['vein']: final_3d_mask[m] = 3
                        for m in st.session_state.masks['tumor']: final_3d_mask[m] = 2
                        
                        sam_nifti_path = os.path.join(tmpdir, "sam_segmentation.nii.gz")
                        nii_img = sitk.GetImageFromArray(np.transpose(final_3d_mask, (2,1,0)))
                        sitk.WriteImage(nii_img, sam_nifti_path)
                        
                        results, _, _, _, _ = perform_full_analysis(
                            open(sam_nifti_path, "rb").read(), "sam_segmentation.nii.gz", 
                            contour_thickness, contact_range, axis, do_2d, do_3d, do_skeleton
                        )
                        
                        display_resectability_recommendation(results)
                        with st.expander("点击查看详细分析数据"): st.json(results)

                        for key in ['masks', 'points', 'labels']:
                            if key in st.session_state: del st.session_state[key]