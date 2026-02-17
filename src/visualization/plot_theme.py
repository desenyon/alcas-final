"""
Unified plotting theme for ALCAS project
Professional, consistent style for all figures
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# ALCAS Color Palette
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'danger': '#D00000',       # Red
    'active': '#06A77D',       # Green (for active-site)
    'allosteric': '#2E86AB',   # Blue (for allosteric)
    'neutral': '#6C757D',      # Gray
    'light': '#E9ECEF',        # Light gray
    'dark': '#212529'          # Dark gray
}

def set_theme():
    """Apply consistent theme to all plots"""
    
    # Set seaborn style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '.2',
        'grid.color': '.9',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
    })
    
    # Matplotlib settings
    rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Figure
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Axes
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 8,
        
        # Ticks
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.edgecolor': '0.8',
    })

def get_color_palette():
    """Get consistent color palette"""
    return [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
            COLORS['success'], COLORS['danger']]

def format_axis(ax, xlabel=None, ylabel=None, title=None):
    """Apply consistent formatting to axis"""
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)