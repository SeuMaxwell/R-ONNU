# -*- coding: utf-8 -*-
"""
学术期刊绘图风格配置模块

本模块提供符合主流学术期刊要求的 Matplotlib 全局风格配置。
支持单栏 (single-column) 和双栏 (double-column) 两种布局。
"""

import matplotlib as mpl
import matplotlib.pyplot as plt  # 新增: 导入 pyplot 用于创建 Figure 对象

# ============================================================================
#                           物理尺寸常量 (英寸)
# ============================================================================

# 单栏宽度 (适用于 Nature, Science, IEEE 等)
SINGLE_COLUMN_WIDTH_INCH = 3.5  # ≈ 89 mm

# 双栏宽度 (适用于期刊全页宽图像)
DOUBLE_COLUMN_WIDTH_INCH = 7.0  # ≈ 178 mm

# 最大高度限制 (建议不超过 9 英寸)
MAX_HEIGHT_INCH = 9.0

# 默认 DPI 设置
SCREEN_DPI = 300
PRINT_DPI = 300
PUBLICATION_DPI = 600

# ============================================================================
#                           学术期刊风格配置
# ============================================================================

JOURNAL_STYLE = {
    # === 字体设置 ===
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,  # 正文 8pt (保持不变，符合标准)

    # === 数学公式字体修正 (重要) ===
    # 强制数学公式(如单位中的μ)使用与正文相同的字体，而不是斜体或不同的数学字体
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',

    # === 坐标轴标签设置 ===
    'axes.labelsize': 8,
    'axes.labelweight': 'normal',
    'axes.labelpad': 2.0,  # [修改] 默认是4.0，改为2.0让标签更贴近轴，适合小图
    'axes.linewidth': 1.0,  # [修改] 加粗边框，从0.8升至1.0，打印更清晰
    'axes.titlesize': 8,  # 标题通常与正文同大或略大，这里设为8保持一致
    'axes.titlepad': 6,  # 标题与图的距离

    # === 刻度设置 ===
    'xtick.labelsize': 7,  # 刻度 7pt (保持不变，比标签小1号是标准做法)
    'ytick.labelsize': 7,
    'xtick.direction': 'in',  # 刻度朝内 (物理/光学领域标准)
    'ytick.direction': 'in',

    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.major.width': 0.8,  # [修改] 加粗刻度，从0.6升至0.8，避免打印模糊
    'ytick.major.width': 0.8,

    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.minor.width': 0.6,  # [修改] 次刻度设为0.6
    'ytick.minor.width': 0.6,

    'xtick.top': True,  # 上下左右都有刻度是封闭图框的标准
    'ytick.right': True,

    # === 图例 ===
    'legend.fontsize': 7,
    'legend.frameon': False,  # 去掉图例边框，显得更现代
    'legend.loc': 'best',

    # === 保存设置 ===
    'figure.dpi': 300,  # 屏幕显示 DPI
    'savefig.dpi': 300,  # 印刷级 DPI
    'savefig.bbox': 'tight',  # 自动裁剪
    'savefig.pad_inches': 0.1,  # [修改] 从0.02改为0.05，解决"头秃"问题，同时不浪费空间

    # === 其他 ===
    'lines.linewidth': 1.2,  # 曲线稍微加粗一点点，从1.0 -> 1.2
    'lines.markersize': 4,
    'axes.grid': False,
    'image.cmap': 'viridis',
}

E_CMAP = 'inferno'       #turbo,cividis，inferno，viridis / magma


# ============================================================================
#                           核心功能函数
# ============================================================================

def apply_journal_style():
    """应用学术期刊风格配置到 Matplotlib 全局设置。"""
    mpl.rcParams.update(JOURNAL_STYLE)


def get_figure_size(width='double', aspect_ratio=0.618):
    """计算图像尺寸 (底层函数)。"""
    if width == 'single':
        w = SINGLE_COLUMN_WIDTH_INCH
    elif width == 'double':
        w = DOUBLE_COLUMN_WIDTH_INCH
    else:
        w = float(width)
    h = min(w * aspect_ratio, MAX_HEIGHT_INCH)
    return (w, h)


def get_subplot_figsize(n_cols, n_rows=1, width='double', subplot_aspect=1.0, cbar_width_ratio=0.05):
    """计算多子图布局尺寸 (底层函数)。"""
    if width == 'single':
        total_width = SINGLE_COLUMN_WIDTH_INCH
    elif width == 'double':
        total_width = DOUBLE_COLUMN_WIDTH_INCH
    else:
        total_width = float(width)

    subplot_width = total_width * (1 - cbar_width_ratio) / n_cols
    subplot_height = subplot_width * subplot_aspect
    fig_height = subplot_height * n_rows
    return (total_width, min(fig_height, MAX_HEIGHT_INCH))


# ============================================================================
#                           新增：便捷绘图封装函数
# ============================================================================

def create_single_column_figure(aspect_ratio=0.75, **kwargs):
    """
    创建一个符合期刊单栏宽度 (89mm) 的 Figure 对象。

    Parameters
    ----------
    aspect_ratio : float
        高宽比 (Height/Width)，默认 0.75 (适合简单曲线或小图)
    **kwargs :
        传递给 plt.figure() 的其他参数 (如 constrained_layout=True)

    Returns
    -------
    fig : matplotlib.figure.Figure
        配置好尺寸和样式的 Figure 对象
    """
    apply_journal_style()
    w, h = get_figure_size(width='single', aspect_ratio=aspect_ratio)
    return plt.figure(figsize=(w, h), **kwargs)


def create_double_column_figure(aspect_ratio=0.618, **kwargs):
    """
    创建一个符合期刊双栏宽度 (178mm) 的 Figure 对象。
    Parameters
    ----------
    aspect_ratio : float
        高宽比 (Height/Width)，默认 0.618 (黄金分割，适合多子图并排)
    **kwargs :
        传递给 plt.figure() 的其他参数

    Returns
    -------
    fig : matplotlib.figure.Figure
        配置好尺寸和样式的 Figure 对象
    """
    apply_journal_style()
    w, h = get_figure_size(width='double', aspect_ratio=aspect_ratio)
    return plt.figure(figsize=(w, h), **kwargs)


# ============================================================================
#                           保存与复用工具
# ============================================================================

DEFAULT_FIGURE_FORMATS = ("png", "pdf")

def save_figure(fig, output_path_no_ext, formats=DEFAULT_FIGURE_FORMATS,
                dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.02,
                metadata=None, transparent=False):
    """以期刊风格默认参数，将同一张图保存为多个格式（推荐 PNG + PDF）。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        要保存的 Figure。
    output_path_no_ext : str
        不带扩展名的输出路径，例如 ``results/figures/loss_curve``。
    formats : tuple[str]
        需要保存的格式列表，例如 ("png", "pdf")。
    dpi : int
        光栅图（png/jpg/tiff 等）的保存 DPI。矢量格式（pdf/svg/eps）将忽略该参数。
    bbox_inches, pad_inches :
        版式裁剪与边距控制。
    metadata : dict | None
        写入到 PDF/PNG 的元数据（如果后端支持）。
    transparent : bool
        是否透明背景。
    """
    apply_journal_style()

    outputs = []
    for fmt in formats:
        fmt_l = str(fmt).lower().lstrip(".")
        out_path = f"{output_path_no_ext}.{fmt_l}"

        # 矢量格式不强制 dpi（避免部分后端出现不一致行为）
        is_raster = fmt_l in {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
        save_kwargs = dict(bbox_inches=bbox_inches, pad_inches=pad_inches,
                           metadata=metadata, transparent=transparent)

        if is_raster:
            fig.savefig(out_path, dpi=dpi, **save_kwargs)
        else:
            fig.savefig(out_path, **save_kwargs)

        outputs.append(out_path)

    return outputs
