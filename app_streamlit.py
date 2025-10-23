# app_streamlit.py (恢复 SAM 2 点击交互 + streamlit-image-coordinates)
# (已集成新的评分卡、评分逻辑和热力图可视化)

import os
import tempfile
import shutil
import time
import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import SimpleITK as sitk

# --- !!! 新增：用于图像点击交互 !!! ---
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    IMAGE_COORDS_AVAILABLE = True
except ImportError:
    st.error("错误：缺少 'streamlit-image-coordinates' 库。")
    st.error("请在您的环境中运行: pip install streamlit-image-coordinates")
    IMAGE_COORDS_AVAILABLE = False
# --- 结束新增 ---

# --- PyTorch 相关导入 ---
import torch
# import torch.nn as nn # DODnet 不再直接在此文件使用

# --- Font Configuration ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Project Modules ---
import toolkit_main as tkm
import toolkit_3D as tk3
from contact3D import calculate_3D_contact
from contact2D import calculate_2D_contact
import sam_segmenter
# --- !!! 移除 import dodnet_inference 或相关代码 !!! ---

# --- !!! 移除 DODnet 模型架构导入 !!! ---
# try:
#     from net import NestedUNet
#     MODEL_ARCH_AVAILABLE = True
# except ImportError:
#     # 不再需要报错，因为我们不用 DODnet 了
#     MODEL_ARCH_AVAILABLE = False


# --- 3D Rendering Imports ---
import plotly.graph_objects as go
try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
    _skimage_import_error = None
except ImportError as e:
    SKIMAGE_AVAILABLE = False
    _skimage_import_error = e

# --- Skeleton Analysis Import ---
try:
    from SkeletonAnalysis import skeleton_analysis, get_skeleton_img
    SKELETON_AVAILABLE = True
    _skeleton_import_error = None
except ImportError as e:
    SKELETON_AVAILABLE = False
    _skeleton_import_error = e

# --- Page Config (必须是第一个 Streamlit 命令) ---
st.set_page_config(page_title="胰腺癌可切除性分析", layout="wide")


# --- 显示导入错误 (在 set_page_config 之后) ---
if _skimage_import_error:
    st.sidebar.error(f"`scikit-image` 未安装 ({_skimage_import_error})。3D 渲染将不可用。")
if _skeleton_import_error:
    st.sidebar.warning(f"无法导入 SkeletonAnalysis ({_skeleton_import_error})。骨架分析功能将不可用。")


# --- Page Title ---
st.title("胰腺癌可切除性分析工具")


# ==============================================================================
# --- !!! 移除 DODnet 相关代码 !!! ---
# ==============================================================================
# class ToTensor(...): # 移除
# class Normalization(...): # 移除
# def load_dodnet_model_internal(...): # 移除
# def run_dodnet_inference(...): # 移除 (或保留为空函数，但不被调用)


# ==============================================================================
# --- !!! 新增：tk3.get_nii 兼容性补丁 (来自用户) !!! ---
# ==============================================================================
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
    # If monkeypatching fails for some reason, continue
    pass


# ==============================================================================
# --- Cached Functions for Performance (修改) ---
# ==============================================================================
@st.cache_data(show_spinner=False)
def perform_full_analysis(_uploaded_file_bytes, _file_name, _contour_thickness, _contact_range, _axis, _do_2d, _do_3d, _do_skeleton, _raw_ct_bytes=None):
    # ... (此函数前半部分保持不变) ...
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, _file_name)
        with open(file_path, "wb") as f: f.write(_uploaded_file_bytes)
        file_dict = {"img_id": os.path.splitext(_file_name)[0], "img_path": file_path, "img_contact_path": None}

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

        label_map_dict_data = tk3.get_nii(file_dict["img_path"], axis=_axis)

        raw_ct_data = None
        if _raw_ct_bytes:
            raw_path = os.path.join(tmpdir, "raw_ct.nii.gz")
            with open(raw_path, "wb") as f: f.write(_raw_ct_bytes)
            raw_ct_data = tk3.get_any_nii(raw_path, axis=_axis)['img']

        contact_img_data = None
        contact_map_path = file_dict.get("img_contact_path")
        if contact_map_path and os.path.exists(contact_map_path):
            try:
                contact_img_data = tk3.get_any_nii(contact_map_path, axis=_axis)['img']
            except Exception as e:
                st.warning(f"无法加载接触面图像: {e}")

        progress_bar.progress(1.0, text="分析完成！")

        # --- !!! 新增：评分逻辑 (来自用户) !!! ---
        # tumor info
        tumor_volume = None
        try:
            # 使用已加载的 label_map_dict_data
            if isinstance(label_map_dict_data, dict) and "tumor" in label_map_dict_data:
                tumor_mask = label_map_dict_data["tumor"]
                tumor_volume = int(np.sum(np.asarray(tumor_mask) > 0))
            elif isinstance(label_map_dict_data, dict) and "origin" in label_map_dict_data:
                # 回退到 'origin' 
                lbl = label_map_dict_data["origin"]
                tumor_volume = int(np.sum(lbl == 2)) # 标签 2 = 肿瘤
            else:
                st.warning("无法在已加载的数据中找到 'tumor' 或 'origin' 键来计算肿瘤体积。")
                
        except Exception as e:
            st.error(f"计算肿瘤体积时出错: {e}")
            tumor_volume = None

        # simple resectability scoring rule
        score = 0.5
        if "artery" in results["3D"] and isinstance(results["3D"]["artery"], list) and len(results["3D"]["artery"]) > 0:
            c3a = results["3D"]["artery"][0].get("contact_ratio", 0)
            score -= 0.3 * c3a
        if "vein" in results["3D"] and isinstance(results["3D"]["vein"], list) and len(results["3D"]["vein"]) > 0:
            c3v = results["3D"]["vein"][0].get("contact_ratio", 0)
            score -= 0.15 * c3v
        if tumor_volume is not None:
            score -= 0.05 * np.log1p(tumor_volume)
        score = float(max(0.0, min(1.0, score)))

        if score > 0.7:
            label = "可能可切除"
        elif score > 0.4:
            label = "边界性"
        else:
            label = "可能不可切除"
        # --- 结束新增评分逻辑 ---

        # --- !!! 修改：返回新增的 score 和 label !!! ---
        return results, label_map_dict_data, raw_ct_data, artery_skeleton, vein_skeleton, contact_img_data, score, label

# ==============================================================================
# --- !!! 替换：使用新的评分卡 (来自用户) !!! ---
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
        reasons.append(f"**主要动脉包裹**: 肿瘤与动脉的最大接触比例为 **{artery_contact_ratio:.2%}**，超过了 180° 包裹的阈值 ({UNRESECTABLE_ARTERY_THRESHOLD:.0%})。")
    elif vein_contact_ratio > BORDERLINE_VEIN_THRESHOLD:
        recommendation = "🟡 **交界可切除 (Borderline Resectable)**"
        reasons.append(f"**主要静脉包裹**: 肿瘤与静脉的最大接触比例为 **{vein_contact_ratio:.2%}**，超过了 180° 包裹的阈值 ({BORDERLINE_VEIN_THRESHOLD:.0%})。")
    elif artery_contact_ratio > 0:
        recommendation = "🟡 **交界可切除 (Borderline Resectable)**"
        reasons.append(f"**动脉邻接**: 肿瘤与动脉存在接触（最大比例 **{artery_contact_ratio:.2%}**），但未达到完全包裹的程度。")
    else:
        reasons.append("肿瘤与主要动脉无接触，且与主要静脉的接触未达到完全包裹的程度，具备良好的手术切除条件。")

    st.markdown(f"### 评估结果: {recommendation}")
    with st.container():
        st.markdown("**评估依据:**")
        for r in reasons: st.markdown(f"- {r}")
        st.markdown(f"**关键参数:**")
        st.markdown(f"  - **动脉最大接触比例**: `{artery_contact_ratio:.2%}`")
        st.markdown(f"  - **静脉最大接触比例**: `{vein_contact_ratio:.2%}`")
        st.caption("注：该建议基于 3D 接触比例。此结果仅供参考。")



def display_score_card(score, label):
    st.markdown("### 切除性评估")
    score_color = "#FF4B4B" if score < 0.4 else ("#FFA500" if score < 0.7 else "#2ECC71")
    st.markdown(f"""
    <div style="border-left: 5px solid {score_color}; padding: 10px; background: #F8F9FA;">
        <p style="font-size: 16px; margin: 0;">评分: <span style="font-weight: bold; color: {score_color}; font-size: 24px;">{score:.2f}</span></p>
        <p style="font-size: 14px; margin: 0;">结论: {label}</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# --- 3D Surface Plotting Function (保持不变) ---
# ==============================================================================
@st.cache_data(show_spinner="正在生成 3D 模型...")
def make_3d_surface_plot(_label_data_array):
    if not SKIMAGE_AVAILABLE:
        st.error("无法创建 3D 视图：缺少 `scikit-image` 库。请运行 `pip install scikit-image`。")
        return go.Figure()

    plot_data = []
    organ_defs = {
        '动脉 (Artery)':   {'label': 1, 'color': 'red',     'opacity': 1.0},
        '肿瘤 (Tumor)':    {'label': 2, 'color': 'green',   'opacity': 0.5},
        '静脉 (Vein)':     {'label': 3, 'color': 'blue',    'opacity': 1.0},
        '胰腺 (Pancreas)': {'label': 4, 'color': 'yellow',  'opacity': 0.4}
    }

    for organ, props in organ_defs.items():
        label_val = props['label']
        if np.any(_label_data_array == label_val):
            try:
                verts, faces, _, _ = marching_cubes(
                    _label_data_array == label_val,
                    level=0.5,
                    spacing=(1.0, 1.0, 1.0)
                )
                plot_data.append(
                    go.Mesh3d(
                        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        color=props['color'],
                        opacity=props['opacity'],
                        name=organ
                    )
                )
            except Exception as e:
                st.warning(f"为 {organ} 生成 3D 网格时出错: {e}")

    if not plot_data:
        st.warning("在 NIfTI 文件中未找到可渲染的标签 (1, 2, 3, 4)。")
        return go.Figure()

    fig = go.Figure(data=plot_data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title="交互式 3D 模型 (可拖动旋转)"
    )
    return fig

# ==============================================================================
# --- 2D Visualization Functions (保持不变) ---
# ==============================================================================
# ... (make_organ_overlay_fig, make_contact_overlay_fig, make_skeleton_fig 函数不变) ...
def make_organ_overlay_fig(label_map_dict, raw_ct_data, axis='z', slice_index=None):
    if axis == 'z':
        slice_idx = slice_index if slice_index is not None else label_map_dict["origin"].shape[2] // 2
        labels = label_map_dict["origin"][:, :, slice_idx]
        raw_slice = raw_ct_data[:, :, slice_idx] if raw_ct_data is not None else None
    else:
        slice_idx = slice_index if slice_index is not None else label_map_dict["origin"].shape[0] // 2
        labels = label_map_dict["origin"][slice_idx, :, :]
        raw_slice = raw_ct_data[slice_idx, :, :] if raw_ct_data is not None else None

    fig, ax = plt.subplots(figsize=(10, 10))

    if raw_slice is not None:
        window_center, window_width = 40, 40
        vmin, vmax = window_center - window_width / 2, window_center + window_width / 2
        ax.imshow(raw_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    else:
        ax.imshow(np.zeros_like(labels.T), cmap='gray', origin='lower')

    colors = {'artery':(1,0,0,.6),'vein':(0,0,1,.6),'tumor':(0,1,0,.6),'pancreas':(1,1,0,.5)}

    for organ, label_val in [('pancreas', 4), ('artery', 1), ('vein', 3), ('tumor', 2)]:
        # Check existence using .get for label_map_dict keys
        # For label_val != 4, we assume they might exist in origin
        if (organ == 'pancreas' and label_map_dict.get(organ) is not None) or \
           (label_val != 4 and label_map_dict.get("origin") is not None):
             try:
                 mask = (labels == label_val).T
                 colored_mask = np.zeros(mask.shape + (4,)); colored_mask[mask] = colors[organ]
                 ax.imshow(colored_mask, origin='lower', interpolation='none')
             except IndexError: # Handle case where slice_idx might be out of bounds somehow
                 pass


    ax.set_aspect('equal')

    legend_elements = [Patch(facecolor=c, label=f"{n} ({l})") for n, c, l in [('动脉', 'red', 'Arteries'), ('静脉', 'blue', 'Veins'), ('肿瘤', 'green', 'Tumor'), ('胰腺', 'yellow', 'Pancreas')]]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(f"器官分割透视图 - 切片 {slice_idx} (轴={axis})"); ax.axis('off')
    return fig

def make_contact_overlay_fig(origin_data, contact_data, axis='z', slice_index=None):
    if axis == 'z':
        slice_idx = slice_index if slice_index is not None else origin_data.shape[2] // 2
        slice_idx = min(slice_idx, origin_data.shape[2] - 1)
        slice_idx = min(slice_idx, contact_data.shape[2] - 1) if contact_data is not None else slice_idx # Handle None contact_data
        o = origin_data[:, :, slice_idx]
        c = contact_data[:, :, slice_idx] if contact_data is not None else np.zeros_like(o) # Handle None contact_data
    else:
        slice_idx = slice_index if slice_index is not None else origin_data.shape[0] // 2
        slice_idx = min(slice_idx, origin_data.shape[0] - 1)
        slice_idx = min(slice_idx, contact_data.shape[0] - 1) if contact_data is not None else slice_idx
        o = origin_data[slice_idx, :, :]
        c = contact_data[slice_idx, :, :] if contact_data is not None else np.zeros_like(o)


    base = np.zeros((o.shape[1], o.shape[0], 3), dtype=np.float32)

    tumor_mask = (o == 2).T
    artery_contour = ((c == 2) | (c == 4)).T
    artery_contact = ((c == 3) | (c == 5)).T
    vein_contour = (c == 2).T # Note: Same label as artery contour in original logic
    vein_contact = (c == 3).T # Note: Same label as artery contact in original logic

    # Consider adjusting the logic if artery/vein contact/contour labels differ
    base[..., 0] = tumor_mask.astype(float) * 0.8
    base[..., 1] = artery_contour.astype(float) * 0.6 + vein_contact.astype(float) * 0.4
    base[..., 2] = artery_contact.astype(float) * 0.8 + vein_contour.astype(float) * 0.3

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(base, origin='lower')
    ax.set_aspect('equal')

    legend_elements = [
        Patch(facecolor=(0.8, 0, 0), label='肿瘤'),
        Patch(facecolor=(0, 0.6, 0.3), label='动脉/静脉轮廓  '),
        Patch(facecolor=(0, 0.4, 0.8), label='动脉/静脉接触  ')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(f"接触面透视图 - 切片 {slice_idx} (轴={axis})")
    ax.axis('off')
    return fig

def make_skeleton_fig(skeleton_img_3d, title, axis='z', slice_index=None):
    # Ensure slice_index is valid
    if axis == 'z':
        if slice_index is None: slice_index = skeleton_img_3d.shape[2] // 2
        slice_index = min(max(0, slice_index), skeleton_img_3d.shape[2] - 1) # Clamp index
        slice_data = skeleton_img_3d[:, :, slice_index]
    else: # axis == 'x'
        if slice_index is None: slice_index = skeleton_img_3d.shape[0] // 2
        slice_index = min(max(0, slice_index), skeleton_img_3d.shape[0] - 1) # Clamp index
        slice_data = skeleton_img_3d[slice_index, :, :]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(slice_data.T, cmap='hot', origin='lower')
    ax.set_aspect('equal')
    ax.set_title(f"{title} - 切片 {slice_index} (轴={axis})"); ax.axis('off')
    return fig

# ==============================================================================
# --- Helper Functions (保持不变) ---
# ==============================================================================
# ... (save_uploaded_file 函数不变) ...
def save_uploaded_file(uploaded_file, directory):
    try:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue()) # Use getvalue() for BytesIO
        return file_path
    except Exception as e:
        st.error(f"保存文件时出错: {e}")
        return None

# ==============================================================================
# --- Sidebar UI & Main App Logic (保持不变) ---
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

    do_3d_render = st.checkbox("交互式 3D 渲染", value=True)
    if do_3d_render and not SKIMAGE_AVAILABLE:
        st.warning("`scikit-image` 库未安装。3D 渲染将不可用。")
        do_3d_render = False

if mode == '上传已分割文件':
    # ... (模式一代码保持不变) ...
    st.header("上传已分割的 NIfTI 文件")
    st.markdown("标签定义: 1=动脉, 2=肿瘤, 3=静脉。(可选: 4=胰腺)")
    uploaded_file = st.file_uploader("上传分割文件", type=["nii", "nii.gz"])

    if uploaded_file:
        # Get bytes once
        uploaded_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        # --- !!! 修改：接收 score 和 label !!! ---
        results, label_map_dict, raw_ct_data, artery_skeleton, vein_skeleton, contact_img_data, score, label = perform_full_analysis(
            uploaded_bytes,
            file_name,
            contour_thickness,
            contact_range,
            axis,
            do_2d,
            do_3d,
            do_skeleton,
            _raw_ct_bytes=None # No raw CT needed unless user uploads it separately
        )
        display_resectability_recommendation(results)
        with st.expander("点击查看详细分析数据"): st.json(results)        
        # --- !!! 替换：显示新的评分卡 !!! ---


        
        # --- !!! 新增：热力图可视化 (来自用户) !!! ---
        if "3D" in results and isinstance(results["3D"], dict):
            
            try:
                import plotly.express as px
                import pandas as pd
                artery_data = results["3D"].get("artery", [])
                vein_data = results["3D"].get("vein", [])
                
                # 检查数据是否存在，并安全地获取第一个元素的接触比例
                artery_ratio = artery_data[0].get("contact_ratio", 0) if len(artery_data) > 0 else 0
                vein_ratio = vein_data[0].get("contact_ratio", 0) if len(vein_data) > 0 else 0

                data = {
                    "血管类型": ["动脉", "静脉"],
                    "接触比例": [artery_ratio, vein_ratio]
                }
                df = pd.DataFrame(data)

                fig = px.imshow(
                    df.pivot_table(values="接触比例", index=None, columns="血管类型"),
                    labels=dict(x="血管类型", y="", color="接触比例"),
                    color_continuous_scale="Viridis",
                    title="3D 接触比例热力图"
                )
                fig.update_layout(width=500, height=300)
                st.plotly_chart(fig)
            except ImportError:
                st.error("无法生成热力图：缺少 `plotly` 或 `pandas` 库。")
            except Exception as e:
                st.warning(f"3D 分析数据不足或格式错误，无法生成可视化图表: {e}")
        # --- 结束新增热力图 ---

        with st.expander("点击查看详细分析数据"): st.json(results)

        if do_3d_render:
            if label_map_dict and label_map_dict.get("origin") is not None:
                st.header("交互式 3D 可视化")
                st.info("您可以拖动、旋转和缩放 3D 模型。")
                fig_3d = make_3d_surface_plot(label_map_dict["origin"])
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("无法生成 3D 视图：未加载标签数据。")

        st.header("2D 切片可视化")
        slice_index = 0 # Initialize slice_index
        if label_map_dict is not None and label_map_dict.get("origin") is not None:
             # Determine max_slice and default_slice based on available data
             data_shape = label_map_dict["origin"].shape
             if axis == 'z':
                 max_slice = data_shape[2] - 1
                 default_slice = data_shape[2] // 2
             else: # axis == 'x'
                 max_slice = data_shape[0] - 1
                 default_slice = data_shape[0] // 2

             # Ensure slider is only created if max_slice is valid
             if max_slice >= 0:
                  slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice), key="upload_slider")

                  col1, col2 = st.columns(2)
                  with col1:
                      # Pass raw_ct_data if available
                      st.pyplot(make_organ_overlay_fig(label_map_dict, raw_ct_data, axis=axis, slice_index=slice_index))
                  with col2:
                      # Only plot contact if contact_img_data exists
                      if contact_img_data is not None:
                         st.pyplot(make_contact_overlay_fig(label_map_dict["origin"], contact_img_data, axis=axis, slice_index=slice_index))
                      else:
                         st.info("未生成或加载接触面图像。")
             else:
                 st.warning("无法确定有效的切片范围。")

        else:
            st.info("未能加载影像数据，无法进行可视化。")


        if do_skeleton and SKELETON_AVAILABLE:
            st.markdown("---")
            st.subheader("骨架分析")
            # Ensure slice_index is defined from the slider above if plots were shown
            if 'slice_index' in locals() and max_slice >=0 :
                col1_skel, col2_skel = st.columns(2)
                with col1_skel:
                    if st.checkbox("显示动脉骨架") and artery_skeleton is not None:
                        st.pyplot(make_skeleton_fig(artery_skeleton, "动脉骨架", axis, slice_index))
                with col2_skel:
                    if st.checkbox("显示静脉骨架") and vein_skeleton is not None:
                        st.pyplot(make_skeleton_fig(vein_skeleton, "静脉骨架", axis, slice_index))
            else:
                 st.info("无法显示骨架，因为切片索引未确定。")


# ==============================================================================
# --- 模式 2：使用 SAM 2 进行交互式分割 ---
# ==============================================================================
elif mode == '使用 SAM 2 分割':
    st.header("使用 SAM 2 进行交互式分割")
    st.markdown("请上传一个**原始的、未分割的**医学影像文件 (例如 CT 平扫期)。")

    # 检查依赖库
    if not IMAGE_COORDS_AVAILABLE:
        st.error("缺少 'streamlit-image-coordinates' 库，无法进行交互式点击。请先安装。")
        st.stop() # 停止执行此模式

    raw_file = st.file_uploader("上传原始影像文件", type=["nii", "nii.gz"], key="sam_raw_uploader")

    # --- 初始化 Session State ---
    # (确保这些只在第一次加载或文件更改时初始化)
    if 'sam_raw_file_id' not in st.session_state or (raw_file and raw_file.file_id != st.session_state.get('sam_raw_file_id')):
        st.session_state.masks = {'artery': [], 'tumor': [], 'vein': []} # 分类存储掩码
        st.session_state.points = [] # 当前切片上的点 [(x, y), ...]
        st.session_state.labels = [] # 当前切片上点的标签 [1, 0, ...] (1=前景, 0=背景)
        st.session_state.current_click_coords = None # 存储最近一次点击的坐标
        st.session_state.raw_img_data = None
        st.session_state.raw_img_nii = None
        st.session_state.normalized_slices = {} # 缓存归一化后的切片
        st.session_state.sam_raw_file_id = raw_file.file_id if raw_file else None
        st.session_state.analysis_complete = False # 标记分析是否完成
        # --- !! 新增：从侧边栏获取参数并存入 state !! ---
        # (确保在点击 "开始分析" 时能获取到最新的侧边栏值)
        st.session_state.sidebar_params = {}

    if raw_file and st.session_state.raw_img_data is None:
         # 只在第一次上传或更换文件时加载
         with st.spinner("正在加载原始 CT 影像..."):
             with tempfile.TemporaryDirectory() as tmpdir:
                 raw_path = save_uploaded_file(raw_file, tmpdir)
                 try:
                     st.session_state.raw_img_nii = nib.load(raw_path)
                     st.session_state.raw_img_data = st.session_state.raw_img_nii.get_fdata().astype(np.float32)
                     st.session_state.normalized_slices = {} # 清空缓存
                     st.rerun() # 重新运行以更新界面状态
                 except Exception as e:
                     st.error(f"加载 NIfTI 文件失败: {e}")
                     st.session_state.raw_img_data = None # 标记加载失败


    if st.session_state.raw_img_data is not None and not st.session_state.analysis_complete:
        raw_img_data = st.session_state.raw_img_data
        H, W, Z = raw_img_data.shape

        col1, col2 = st.columns([3, 1]) # 图像列更宽

# ... (代码上文) ...

        with col1:
            st.subheader("图像交互区域")
            slice_idx = st.slider("选择要标注的切片", 0, Z - 1, Z // 2, key="sam_slice_slider")

            # --- 缓存和获取当前切片 (归一化到 uint8 用于显示) ---
            if slice_idx not in st.session_state.normalized_slices:
                current_slice_raw = raw_img_data[:, :, slice_idx] # (H, W)
                # ... (归一化逻辑不变) ...
                slice_normalized = np.clip(current_slice_raw, -100, 400)
                min_norm, max_norm = np.min(slice_normalized), np.max(slice_normalized)
                if max_norm > min_norm:
                    slice_uint8 = ((slice_normalized - min_norm) / (max_norm - min_norm) * 255).astype(np.uint8)
                else:
                    slice_uint8 = np.zeros_like(current_slice_raw, dtype=np.uint8)
                # 存储原始 (H, W) 切片
                st.session_state.normalized_slices[slice_idx] = slice_uint8
            
            current_slice_uint8 = st.session_state.normalized_slices[slice_idx] # (H, W)
            
            # --- !!! 修改：创建用于显示的转置 (W, H) 图像 ---
            # 这将用于 clicker 和 matplotlib，以确保它们一致
            display_slice = current_slice_uint8.T  # (W, H)
            # --- 结束修改 ---

            # --- 使用 streamlit-image-coordinates 进行点击交互 ---
            st.write("在下方图像上点击选择点：")
            
            # --- !!! 修改：使用 display_slice (W, H) ---
            value = streamlit_image_coordinates(display_slice, key="sam_image_click")
            # --- 结束修改 ---

            # --- !!! 修改：点击逻辑 ---
            # 现在的 (x, y) 坐标是相对于 (W, H) 图像的
            # x 对应 H 轴, y 对应 W 轴
            if value is not None and value != st.session_state.get("_last_click_value_ref"):
                coords = (value["x"], value["y"])
                st.session_state.current_click_coords = coords
                st.session_state._last_click_value_ref = value 
                st.rerun() 
            # --- 结束点击逻辑修改 ---


            # --- 使用 Matplotlib 显示带有点的图像 ---
            fig, ax = plt.subplots(figsize=(8, 8)) 
            
            # --- !!! 修改：使用 display_slice (W, H) ---
            # 这与之前的 .T 效果相同，满足“不要旋转”的要求
            ax.imshow(display_slice, cmap='gray', origin='lower')
            # --- 结束修改 ---

            ax.set_title(f"当前切片: {slice_idx} (已添加 {len(st.session_state.points)} 个点)")
            ax.set_axis_off()
            
            # --- 坐标绘制 (无需修改) ---
            # 因为 clicker 和 imshow 图像一致，
            # (x, y) 点 (point[0], point[1]) 会被正确绘制
            # point[0] (x) -> H 轴, point[1] (y) -> W 轴
            for i, (point, label) in enumerate(zip(st.session_state.points, st.session_state.labels)):
                color = 'green' if label == 1 else 'red'
                marker = '+' if label == 1 else 'x'
                ax.scatter(point[0], point[1], color=color, marker=marker, s=150, linewidths=3)
                ax.text(point[0] + 5, point[1] + 5, str(i+1), color=color, fontsize=12) 
            
            if st.session_state.current_click_coords:
               ax.scatter(st.session_state.current_click_coords[0], st.session_state.current_click_coords[1], 
                          color='yellow', marker='*', s=200, linewidths=2, edgecolors='black')
            
            ax.set_aspect('equal') 
            st.pyplot(fig)
            # --- 结束 Matplotlib 显示 ---

        with col2:
            st.subheader("标注工具")
            structure_to_label = st.radio("选择要标注的结构", ('肿瘤', '动脉', '静脉'), key="sam_structure_radio")
            label_map = {'肿瘤': 'tumor', '动脉': 'artery', '静脉': 'vein'}

            # --- 坐标信息 (无需修改) ---
            if st.session_state.current_click_coords:
                st.info(f"待确认点: ({st.session_state.current_click_coords[0]}, {st.session_state.current_click_coords[1]})")
            else:
                st.info("请在左侧图像上点击一个点。")

            # --- 添加点按钮 (无需修改) ---
            # 存储 (x, y) 坐标，现在它们对应 (H_coord, W_coord)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("添加前景点 (+)", key="sam_add_fg"):
                    if st.session_state.current_click_coords:
                        st.session_state.points.append(st.session_state.current_click_coords)
                        # ... (st.rerun() 等)
                        st.session_state.labels.append(1)
                        st.session_state.current_click_coords = None 
                        st.session_state._last_click_value_ref = None 
                        st.rerun()
                    else:
                        st.warning("请先在图像上点击选择一个点。")
            with col_btn2:
                 if st.button("添加背景点 (-)", key="sam_add_bg"):
                    if st.session_state.current_click_coords:
                        st.session_state.points.append(st.session_state.current_click_coords)
                        st.session_state.labels.append(0) 
                        st.session_state.current_click_coords = None
                        st.session_state._last_click_value_ref = None
                        st.rerun()
                    else:
                        st.warning("请先在图像上点击选择一个点。")

            # --- 其他控制按钮 (无需修改) ---
            if st.button("清除当前切片所有点", key="sam_clear_points"):
                st.session_state.points = []
                st.session_state.labels = []
                st.session_state.current_click_coords = None
                st.session_state._last_click_value_ref = None
                st.rerun()

            if st.button("运行 SAM 分割当前切片", key="sam_run_slice_seg"):
                if not st.session_state.points:
                    st.warning("请至少添加一个前景或背景点。")
                else:
                    with st.spinner("SAM 正在分割当前切片..."):
                        predictor = sam_segmenter.load_sam2_model() 
                        if predictor:
                            
                            # --- !!! 修改：准备 SAM 的输入图像 ---
                            # 我们需要 (W, H) 图像，即 display_slice
                            # 确保我们从缓存中获取 (H, W) 并转置它
                            current_slice_hw_uint8 = st.session_state.normalized_slices[slice_idx].T 
                            # (上面的 current_slice_for_sam_hw 和 ...uint8 都不需要了)
                            # --- 结束修改 ---
                            
                            # --- !!! 修改：调用 SAM ---
                            # 图像是 (W, H)，点是 (H_coord, W_coord)，这现在是匹配的
                            mask_hw = sam_segmenter.run_sam2_prediction(
                                predictor,
                                current_slice_hw_uint8, # 传递 (W, H) 图像
                                st.session_state.points,  # 传递 (H_coord, W_coord) 坐标列表
                                st.session_state.labels
                            )
                            # --- 结束修改 ---

                            if mask_hw is not None:
                                # mask_hw 是 (W, H)
                                target_key = label_map[structure_to_label]
                                full_mask = np.zeros_like(raw_img_data, dtype=bool) # (H, W, Z)
                                
                                # --- !!! 修改：将 (W, H) 掩码转置回 (H, W) ---
                                full_mask[:, :, slice_idx] = mask_hw.T
                                # --- 结束修改 ---

                                st.session_state.masks[target_key].append(full_mask)

                                st.success(f"已为“{structure_to_label}”添加一个掩码 (来自切片 {slice_idx})。")
                                # ... (清除点并 rerun)
                                st.session_state.points = []
                                st.session_state.labels = []
                                st.session_state.current_click_coords = None
                                st.session_state._last_click_value_ref = None
                                st.rerun()
                            else:
                                st.error("SAM 分割失败，请检查点或图像。")
                        else:
                             st.error("SAM 模型加载失败，无法进行分割。")

# ... (代码下文) ...

        # --- 显示已完成的掩码数量 ---
        st.markdown("---")
        st.subheader("已完成分割")
        st.write("已添加的 3D 掩码数量:")
        st.write(f"- 肿瘤: {len(st.session_state.masks['tumor'])} 个")
        st.write(f"- 动脉: {len(st.session_state.masks['artery'])} 个")
        st.write(f"- 静脉: {len(st.session_state.masks['vein'])} 个")
        st.caption("每个掩码代表一次成功的切片分割。")

        # --- 完成与分析按钮 ---
        if st.button("完成所有分割，合并掩码并开始分析", key="sam_finalize"):
            if not any(st.session_state.masks.values()):
                st.error("请至少完成一个结构的分割。")
            else:
                with st.spinner("正在合并所有分割掩码..."):
                    # 合并同一结构的所有 3D 掩码
                    final_3d_mask = np.zeros_like(raw_img_data, dtype=np.uint8)
                    for target, label_val in [('artery', 1), ('tumor', 2), ('vein', 3)]:
                        combined_mask_for_target = np.zeros_like(raw_img_data, dtype=bool)
                        for mask_3d in st.session_state.masks[target]:
                            combined_mask_for_target = np.logical_or(combined_mask_for_target, mask_3d)
                        final_3d_mask[combined_mask_for_target] = label_val

                with tempfile.TemporaryDirectory() as tmpdir:
                     # 保存合并后的掩码为 NIfTI
                     sam_nifti_path = os.path.join(tmpdir, "sam_merged_segmentation.nii.gz")
                     # 使用原始 NII 的仿射矩阵和头信息
                     refined_nii = nib.Nifti1Image(final_3d_mask.astype(np.uint8), st.session_state.raw_img_nii.affine, st.session_state.raw_img_nii.header)
                     nib.save(refined_nii, sam_nifti_path)

                     with open(sam_nifti_path, "rb") as f:
                         sam_nifti_bytes = f.read()

                     st.success("掩码合并完成，准备进行后续分析。")

                     # --- 调用分析函数 ---
                     st.info("正在运行接触分析和可视化...")
                     
                     # --- !!! 修改：从侧边栏全局变量获取最新参数 !!! ---
                     # (假设侧边栏变量是全局可访问的)
                     current_contour_thickness = contour_thickness
                     current_contact_range = contact_range
                     current_axis = axis
                     current_do_2d = do_2d
                     current_do_3d = do_3d
                     current_do_skeleton = do_skeleton
                     current_do_3d_render = do_3d_render
                     
                     # --- !!! 修改：接收 score 和 label !!! ---
                     results, label_map_dict, raw_ct_data_final, artery_skeleton, vein_skeleton, contact_img_data, score, label = perform_full_analysis(
                         sam_nifti_bytes,
                         "sam_merged_segmentation.nii.gz",
                         current_contour_thickness,
                         current_contact_range,
                         current_axis,
                         current_do_2d,
                         current_do_3d,
                         current_do_skeleton,
                         _raw_ct_bytes=raw_file.getvalue() # 传递原始 CT 数据用于显示
                     )
                     
                     # --- !!! 修改：在 session_state 中保存 score 和 label !!! ---
                     st.session_state.analysis_results = (
                         results, label_map_dict, raw_ct_data_final, 
                         artery_skeleton, vein_skeleton, contact_img_data, 
                         score, label # <-- 新增
                     )
                     st.session_state.analysis_axis = current_axis # 保存用于可视化的轴
                     st.session_state.analysis_do_skeleton = current_do_skeleton
                     st.session_state.analysis_do_3d_render = current_do_3d_render

                     # 标记分析完成并重新运行以显示结果
                     st.session_state.analysis_complete = True
                     st.rerun()


    # --- 在分析完成后显示结果 ---
    if st.session_state.get('analysis_complete', False):
        
        # --- !!! 修改：从 session_state 中解包 score 和 label !!! ---
        results, label_map_dict, raw_ct_data_final, artery_skeleton, vein_skeleton, contact_img_data, score, label = st.session_state.analysis_results
        
        axis = st.session_state.analysis_axis
        do_skeleton = st.session_state.analysis_do_skeleton
        do_3d_render = st.session_state.analysis_do_3d_render


        st.header("SAM 2 交互分割后的分析结果")
        
        # --- !!! 替换：显示新的评分卡 !!! ---
        display_resectability_recommendation(results)
        with st.expander("点击查看详细分析数据"):
            st.json(results)
        
        # --- !!! 新增：热力图可视化 (来自用户) !!! ---
        if "3D" in results and isinstance(results["3D"], dict):
            
            try:
                import plotly.express as px
                import pandas as pd
                artery_data = results["3D"].get("artery", [])
                vein_data = results["3D"].get("vein", [])
                
                # 检查数据是否存在，并安全地获取第一个元素的接触比例
                artery_ratio = artery_data[0].get("contact_ratio", 0) if len(artery_data) > 0 else 0
                vein_ratio = vein_data[0].get("contact_ratio", 0) if len(vein_data) > 0 else 0

                data = {
                    "血管类型": ["动脉", "静脉"],
                    "接触比例": [artery_ratio, vein_ratio]
                }
                df = pd.DataFrame(data)

                fig = px.imshow(
                    df.pivot_table(values="接触比例", index=None, columns="血管类型"),
                    labels=dict(x="血管类型", y="", color="接触比例"),
                    color_continuous_scale="Viridis",
                    title="3D 接触比例热力图"
                )
                fig.update_layout(width=500, height=300)
                st.plotly_chart(fig)
            except ImportError:
                st.error("无法生成热力图：缺少 `plotly` 或 `pandas` 库。")
            except Exception as e:
                st.warning(f"3D 分析数据不足或格式错误，无法生成可视化图表: {e}")
        # --- 结束新增热力图 ---
        
        with st.expander("点击查看详细分析数据"):
            st.json(results)

        if do_3d_render:
            if label_map_dict and label_map_dict.get("origin") is not None:
                st.header("交互式 3D 可视化")
                st.info("您可以拖动、旋转和缩放 3D 模型。")
                fig_3d = make_3d_surface_plot(label_map_dict["origin"])
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("无法生成 3D 视图：未加载标签数据。")

        st.header("2D 切片可视化")
        if label_map_dict and raw_ct_data_final is not None:
            data_shape = label_map_dict["origin"].shape
            if axis == 'z':
                max_slice = data_shape[2] - 1
                default_slice = data_shape[2] // 2
            else: # axis == 'x'
                max_slice = data_shape[0] - 1
                default_slice = data_shape[0] // 2

            if max_slice >=0:
                slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice), key="sam_result_slider")

                col1_res, col2_res = st.columns(2)
                with col1_res:
                    st.pyplot(make_organ_overlay_fig(label_map_dict, raw_ct_data_final, axis=axis, slice_index=slice_index))
                with col2_res:
                    if contact_img_data is not None:
                        st.pyplot(make_contact_overlay_fig(label_map_dict["origin"], contact_img_data, axis=axis, slice_index=slice_index))
                    else:
                        st.info("未生成或加载接触面图像。")
            else:
                 st.warning("无法确定有效的切片范围用于结果可视化。")

        else:
            st.info("未能加载SAM分割影像或原始CT影像，无法进行可视化。")

        if do_skeleton and SKELETON_AVAILABLE:
            st.markdown("---")
            st.subheader("骨架分析")
            if 'slice_index' in locals() and max_slice >=0: # Check if slider was created
                col1_skel_res, col2_skel_res = st.columns(2)
                with col1_skel_res:
                    if st.checkbox("显示动脉骨架", key="sam_res_skel_art") and artery_skeleton is not None:
                        st.pyplot(make_skeleton_fig(artery_skeleton, "动脉骨架", axis, slice_index))
                with col2_skel_res:
                    if st.checkbox("显示静脉骨架", key="sam_res_skel_vein") and vein_skeleton is not None:
                        st.pyplot(make_skeleton_fig(vein_skeleton, "静脉骨架", axis, slice_index))
            else:
                st.info("无法显示骨架，因为切片索引未确定。")

        # --- 添加按钮以开始新的 SAM 分割 ---
        if st.button("开始新的 SAM 交互分割", key="sam_restart"):
             # 清理 session state 以重新开始
             keys_to_reset = ['masks', 'points', 'labels', 'current_click_coords',
                              'raw_img_data', 'raw_img_nii', 'normalized_slices',
                              'sam_raw_file_id', 'analysis_complete', 'analysis_results',
                              'analysis_axis', 'analysis_do_skeleton', 'analysis_do_3d_render',
                              '_last_click_value_ref', 'sidebar_params'] # <-- 也清空 sidebar_params
             for key in keys_to_reset:
                 if key in st.session_state:
                     del st.session_state[key]
             # 可能还需要清除上传组件的状态，重新运行通常可以做到
             st.rerun()