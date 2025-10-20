# app_streamlit.py (Fixed NameError in make_contact_overlay_fig)

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

# --- Font Configuration ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- Project Modules ---
import toolkit_main as tkm
import toolkit_3D as tk3
from contact3D import calculate_3D_contact
from contact2D import calculate_2D_contact
import sam_segmenter

# --- NEW 3D Rendering Imports ---
import plotly.graph_objects as go
try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    st.sidebar.error("`scikit-image` 未安装。3D 渲染将不可用。")
    SKIMAGE_AVAILABLE = False
# --- End 3D Imports ---

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
def perform_full_analysis(_uploaded_file_bytes, _file_name, _contour_thickness, _contact_range, _axis, _do_2d, _do_3d, _do_skeleton, _raw_ct_bytes=None):
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
        return results, label_map_dict_data, raw_ct_data, artery_skeleton, vein_skeleton, contact_img_data

# ==============================================================================
# --- Resectability Advisor Module ---
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

# ==============================================================================
# --- 3D Surface Plotting Function ---
# ==============================================================================

@st.cache_data(show_spinner="正在生成 3D 模型...")
def make_3d_surface_plot(_label_data_array):
    """Generates an interactive 3D surface plot using Plotly."""
    
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
# --- 2D Visualization Functions (Corrected Aspect Ratio) ---
# ==============================================================================

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
        window_center, window_width = 40, 400
        vmin, vmax = window_center - window_width / 2, window_center + window_width / 2
        ax.imshow(raw_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    else: 
        ax.imshow(np.zeros_like(labels.T), cmap='gray', origin='lower')

    colors = {'artery':(1,0,0,.6),'vein':(0,0,1,.6),'tumor':(0,1,0,.6),'pancreas':(1,1,0,.5)}
    
    for organ, label_val in [('pancreas', 4), ('artery', 1), ('vein', 3), ('tumor', 2)]:
        if organ in label_map_dict or label_val !=4:
            mask = (labels == label_val).T
            colored_mask = np.zeros(mask.shape + (4,)); colored_mask[mask] = colors[organ]
            ax.imshow(colored_mask, origin='lower', interpolation='none')
            
    ax.set_aspect('equal')
    
    legend_elements = [Patch(facecolor=c, label=f"{n} ({l})") for n, c, l in [('动脉', 'red', 'Arteries'), ('静脉', 'blue', 'Veins'), ('肿瘤', 'green', 'Tumor'), ('胰腺', 'yellow', 'Pancreas')]]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(f"器官分割透视图 - 切片 {slice_idx} (轴={axis})"); ax.axis('off')
    return fig

# --- FIX: Renamed variable to slice_idx to match the title ---
def make_contact_overlay_fig(origin_data, contact_data, axis='z', slice_index=None):
    """(New) Plots Contact Map (Tumor + Contact/Contours)"""
    if axis == 'z':
        slice_idx = slice_index if slice_index is not None else origin_data.shape[2] // 2
        slice_idx = min(slice_idx, origin_data.shape[2] - 1)
        slice_idx = min(slice_idx, contact_data.shape[2] - 1)
        o = origin_data[:, :, slice_idx]
        c = contact_data[:, :, slice_idx]
    else:
        slice_idx = slice_index if slice_index is not None else origin_data.shape[0] // 2
        slice_idx = min(slice_idx, origin_data.shape[0] - 1)
        slice_idx = min(slice_idx, contact_data.shape[0] - 1)
        o = origin_data[slice_idx, :, :]
        c = contact_data[slice_idx, :, :]
    # --- END FIX ---

    base = np.zeros((o.shape[1], o.shape[0], 3), dtype=np.float32)
    
    tumor_mask = (o == 2).T
    artery_contour = ((c == 2) | (c == 4)).T
    artery_contact = ((c == 3) | (c == 5)).T
    vein_contour = (c == 2).T
    vein_contact = (c == 3).T

    base[..., 0] = tumor_mask.astype(float) * 0.8 # Red channel
    base[..., 1] = artery_contour.astype(float) * 0.6 + vein_contact.astype(float) * 0.4 # Green channel
    base[..., 2] = artery_contact.astype(float) * 0.8 + vein_contour.astype(float) * 0.3 # Blue channel

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(base, origin='lower')
    ax.set_aspect('equal')
    
    legend_elements = [
        Patch(facecolor=(0.8, 0, 0), label='肿瘤 (Tumor)'),
        Patch(facecolor=(0, 0.6, 0.3), label='动脉/静脉轮廓 (Vessel Contour)'),
        Patch(facecolor=(0, 0.4, 0.8), label='动脉/静脉接触 (Vessel Contact)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # --- FIX: This line will now work correctly ---
    ax.set_title(f"接触面透视图 - 切片 {slice_idx} (轴={axis})")
    ax.axis('off')
    return fig
# --- END FIX ---


def make_skeleton_fig(skeleton_img_3d, title, axis='z', slice_index=None):
    if axis == 'z':
        if slice_index is None: slice_index = skeleton_img_3d.shape[2] // 2
        slice_data = skeleton_img_3d[:, :, slice_index]
    else:
        if slice_index is None: slice_index = skeleton_img_3d.shape[0] // 2
        slice_data = skeleton_img_3d[slice_index, :, :]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(slice_data.T, cmap='hot', origin='lower')
    ax.set_aspect('equal')
    ax.set_title(f"{title} - 切片 {slice_index} (轴={axis})"); ax.axis('off')
    return fig

def save_uploaded_file(uploaded_file, directory):
    try:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return file_path
    except Exception as e:
        st.error(f"保存文件时出错: {e}")
        return None
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
    
    do_3d_render = st.checkbox("交互式 3D 渲染", value=True)
    if do_3d_render and not SKIMAGE_AVAILABLE:
        st.warning("`scikit-image` 库未安装。3D 渲染将不可用。")
        do_3d_render = False

if mode == '上传已分割文件':
    st.header("上传已分割的 NIfTI 文件")
    st.markdown("标签定义: 1=动脉, 2=肿瘤, 3=静脉。(可选: 4=胰腺)")
    uploaded_file = st.file_uploader("上传分割文件", type=["nii", "nii.gz"])

    if uploaded_file:
        results, label_map_dict, raw_ct_data, artery_skeleton, vein_skeleton, contact_img_data = perform_full_analysis(
            uploaded_file.getvalue(), 
            uploaded_file.name, 
            contour_thickness, 
            contact_range, 
            axis, 
            do_2d, 
            do_3d, 
            do_skeleton,
            _raw_ct_bytes=None 
        )
        display_resectability_recommendation(results)
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
        if label_map_dict is not None and contact_img_data is not None:
            max_slice = label_map_dict["origin"].shape[2] - 1 if axis == 'z' else label_map_dict["origin"].shape[0] - 1
            default_slice = label_map_dict["origin"].shape[2] // 2 if axis == 'z' else label_map_dict["origin"].shape[0] // 2
            slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice), key="upload_slider")
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(make_organ_overlay_fig(label_map_dict, raw_ct_data, axis=axis, slice_index=slice_index))
            with col2:
                st.pyplot(make_contact_overlay_fig(label_map_dict["origin"], contact_img_data, axis=axis, slice_index=slice_index))
        
        elif label_map_dict is not None:
            max_slice = label_map_dict["origin"].shape[2] - 1 if axis == 'z' else label_map_dict["origin"].shape[0] - 1
            default_slice = label_map_dict["origin"].shape[2] // 2 if axis == 'z' else label_map_dict["origin"].shape[0] // 2
            slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice), key="upload_slider_fb")
            st.pyplot(make_organ_overlay_fig(label_map_dict, raw_ct_data, axis=axis, slice_index=slice_index))
        
        else:
            st.info("未能加载影像数据，无法进行可视化。")

        if do_skeleton and SKELETON_AVAILABLE:
            st.markdown("---")
            st.subheader("骨架分析")
            if 'slice_index' not in locals(): 
                slice_index = label_map_dict["origin"].shape[2] // 2 if axis == 'z' else label_map_dict["origin"].shape[0] // 2
                
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("显示动脉骨架") and artery_skeleton is not None: 
                    st.pyplot(make_skeleton_fig(artery_skeleton, "动脉骨架", axis, slice_index))
            with col2:
                if st.checkbox("显示静脉骨架") and vein_skeleton is not None: 
                    st.pyplot(make_skeleton_fig(vein_skeleton, "静脉骨架", axis, slice_index))

elif mode == '使用 SAM 2 分割':
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
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(current_slice.T, cmap='gray', origin='lower')
                ax.set_aspect('equal')
                ax.set_title(f"当前切片: {slice_idx}")
                ax.set_axis_off()
                for point, label in zip(st.session_state.points, st.session_state.labels):
                    color = 'green' if label == 1 else 'red'
                    ax.scatter(point[0], point[1], color=color, marker='*', s=100)
                st.pyplot(fig)
                
                st.info("说明: Streamlit 目前不直接支持图像点击事件。请在下方手动输入坐标来模拟点击。")
            with col2:
                st.subheader("标注工具")
                structure_to_label = st.radio("选择要标注的结构", ('肿瘤', '动脉', '静脉'))
                label_map = {'肿瘤': 'tumor', '动脉': 'artery', '静脉': 'vein'}
                x_coord = st.number_input("输入 X 坐标", 0, current_slice.shape[0]-1, step=1)
                y_coord = st.number_input("输入 Y 坐标", 0, current_slice.shape[1]-1, step=1)
                if st.button("添加前景点 (Positive Point)"):
                    st.session_state.points.append((x_coord, y_coord)); st.session_state.labels.append(1); st.rerun()
                if st.button("运行 SAM 分割当前切片"):
                    predictor = sam_segmenter.load_sam2_model()
                    if predictor and st.session_state.points:
                        with st.spinner("SAM 2 正在分割..."):
                            mask = sam_segmenter.run_sam2_prediction(predictor, current_slice, st.session_state.points, st.session_state.labels)
                            target_key = label_map[structure_to_label]
                            full_mask = np.zeros_like(raw_img_data, dtype=bool); full_mask[:, :, slice_idx] = mask
                            st.session_state.masks[target_key].append(full_mask)
                            st.success(f"已为“{structure_to_label}”添加一个分割掩码。")
                            st.session_state.points = []; st.session_state.labels = []; st.rerun()
                    else: st.warning("请至少添加一个前景点。")
                if st.button("清除当前点"):
                    st.session_state.points = []; st.session_state.labels = []; st.rerun()
            
            st.markdown("---")
            st.subheader("完成与分析")
            st.write("已分割的掩码数量:")
            st.write(f"- 肿瘤: {len(st.session_state.masks['tumor'])} 个")
            st.write(f"- 动脉: {len(st.session_state.masks['artery'])} 个")
            st.write(f"- 静脉: {len(st.session_state.masks['vein'])} 个")

            st.subheader("分析参数")
            sam_contour_thickness = st.slider("轮廓厚度", 0.5, 5.0, 1.5, key="sam_contour")
            sam_contact_range = st.slider("接触范围 (像素)", 0, 10, 2, key="sam_contact")
            sam_axis = st.selectbox("2D 观察轴", ["z", "x"], index=0, key="sam_axis")
            
            st.subheader("分析模块")
            sam_do_2d = st.checkbox("2D 接触分析", value=True, key="sam_2d")
            sam_do_3d = st.checkbox("3D 接触分析", value=True, key="sam_3d")
            sam_do_skeleton = st.checkbox("骨架分析", value=True, key="sam_skeleton")
            if sam_do_skeleton and not SKELETON_AVAILABLE: 
                st.warning("骨架分析模块不可用。")
                sam_do_skeleton = False
            
            sam_do_3d_render = st.checkbox("交互式 3D 渲染", value=True, key="sam_3d_render")
            if sam_do_3d_render and not SKIMAGE_AVAILABLE:
                st.warning("`scikit-image` 库未安装。3D 渲染将不可用。")
                sam_do_3d_render = False

            if st.button("完成所有分割，开始分析"):
                if not any(st.session_state.masks.values()):
                    st.error("请至少完成一个结构的分割。")
                else:
                    with st.spinner("正在合并所有分割掩码并分析..."):
                        final_3d_mask = np.zeros_like(raw_img_data, dtype=np.uint8)
                        for mask_3d in st.session_state.masks['artery']: final_3d_mask[mask_3d] = 1
                        for mask_3d in st.session_state.masks['vein']: final_3d_mask[mask_3d] = 3
                        for mask_3d in st.session_state.masks['tumor']: final_3d_mask[mask_3d] = 2
                        
                        sam_nifti_path = os.path.join(tmpdir, "sam_segmentation.nii.gz")
                        nii_img = sitk.GetImageFromArray(np.transpose(final_3d_mask, (2,1,0)))
                        sitk.WriteImage(nii_img, sam_nifti_path)
                        
                        with open(sam_nifti_path, "rb") as f:
                            sam_nifti_bytes = f.read()
                        
                        st.success("SAM 分割结果已保存，准备进行后续分析。")
                        
                        results, label_map_dict, raw_ct_data_final, artery_skeleton, vein_skeleton, contact_img_data = perform_full_analysis(
                            sam_nifti_bytes, 
                            "sam_segmentation.nii.gz", 
                            sam_contour_thickness, 
                            sam_contact_range, 
                            sam_axis, 
                            sam_do_2d, 
                            sam_do_3d, 
                            sam_do_skeleton,
                            _raw_ct_bytes=raw_file.getvalue()
                        )

                        st.header("SAM 分割后的分析结果")
                        display_resectability_recommendation(results)
                        with st.expander("点击查看详细分析数据"): 
                            st.json(results)
                        
                        if sam_do_3d_render:
                            if label_map_dict and label_map_dict.get("origin") is not None:
                                st.header("交互式 3D 可视化")
                                st.info("您可以拖动、旋转和缩放 3D 模型。")
                                fig_3d = make_3d_surface_plot(label_map_dict["origin"])
                                st.plotly_chart(fig_3d, use_container_width=True)
                            else:
                                st.warning("无法生成 3D 视图：未加载标签数据。")
                            
                        st.header("2D 切片可视化")
                        if label_map_dict and raw_ct_data_final is not None and contact_img_data is not None:
                            max_slice = label_map_dict["origin"].shape[2] - 1 if sam_axis == 'z' else label_map_dict["origin"].shape[0] - 1
                            default_slice = label_map_dict["origin"].shape[2] // 2 if sam_axis == 'z' else label_map_dict["origin"].shape[0] // 2
                            slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice), key="sam_slider")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(make_organ_overlay_fig(label_map_dict, raw_ct_data_final, axis=sam_axis, slice_index=slice_index))
                            with col2:
                                st.pyplot(make_contact_overlay_fig(label_map_dict["origin"], contact_img_data, axis=sam_axis, slice_index=slice_index))
                        
                        elif label_map_dict and raw_ct_data_final is not None:
                            max_slice = label_map_dict["origin"].shape[2] - 1 if sam_axis == 'z' else label_map_dict["origin"].shape[0] - 1
                            default_slice = label_map_dict["origin"].shape[2] // 2 if sam_axis == 'z' else label_map_dict["origin"].shape[0] // 2
                            slice_index = st.slider("选择切片索引", 0, max_slice, min(default_slice, max_slice), key="sam_slider_fb")
                            st.pyplot(make_organ_overlay_fig(label_map_dict, raw_ct_data_final, axis=sam_axis, slice_index=slice_index))

                        else:
                            st.info("未能加载SAM分割影像或原始CT影像，无法进行可视化。")
                            
                        if sam_do_skeleton and SKELETON_AVAILABLE:
                            st.markdown("---")
                            st.subheader("骨架分析")
                            if 'slice_index' not in locals():
                                slice_index = label_map_dict["origin"].shape[2] // 2 if sam_axis == 'z' else label_map_dict["origin"].shape[0] // 2

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.checkbox("显示动脉骨架", key="sam_skel_art") and artery_skeleton is not None: 
                                    st.pyplot(make_skeleton_fig(artery_skeleton, "动脉骨架", sam_axis, slice_index))
                            with col2:
                                if st.checkbox("显示静脉骨架", key="sam_skel_vein") and vein_skeleton is not None: 
                                    st.pyplot(make_skeleton_fig(vein_skeleton, "静脉骨架", sam_axis, slice_index))

                        for key in ['masks', 'points', 'labels']:
                            if key in st.session_state: del st.session_state[key]