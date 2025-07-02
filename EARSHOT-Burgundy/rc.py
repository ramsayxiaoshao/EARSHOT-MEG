# Importing this sets the matplotlib RC parmeters
from pathlib import Path

from eelbrain import plot
from eelbrain.plot._colors import adjust_hls
from matplotlib import pyplot


# Where to save plots
DST = Path('/Users/christian/Library/Mobile Documents/com~apple~CloudDocs/Research/Earshot-Burgundy/Figures/plots')

# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 9
LINEWIDTH = 0.5
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    # line width
    'axes.linewidth': LINEWIDTH,
    'grid.linewidth': LINEWIDTH,
    'lines.linewidth': 1,
    'patch.linewidth': LINEWIDTH,
    'xtick.major.width': LINEWIDTH,
    'xtick.minor.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH,
    'ytick.minor.width': LINEWIDTH,
}
pyplot.rcParams.update(RC)

LOSS = [None] + [f'dw{w}to10' for w in [16, 64, 256, 1024, 4096]]
LOSS_COLORS = {'constant': 'k'}
LOSS_COLORS.update(plot.colors_for_oneway(LOSS[:-1], cmap='viridis'))
LOSS_COLORS.pop(None)
LOSS_COLORS['dw256to10'] = adjust_hls(LOSS_COLORS['dw256to10'], lightness=-0.20)
LOSS_COLORS['dw1024to10'] = '#D55E00'
# LOSS_COLORS['dw1024to10'] = '#E69F00'
LOSS_COLORS['dw4096to10'] = 'red'
