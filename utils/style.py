
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm

def make_default_colormap():

    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    start = viridis(0.5)

    newcolors[:128, 0] = np.linspace(1, start[0], 128)
    newcolors[:128, 1] = np.linspace(1, start[1], 128)
    newcolors[:128, 2] = np.linspace(1, start[2], 128)
    newcmp = ListedColormap(newcolors)

    return newcmp


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


basic_colormap = make_default_colormap()
dark_colormap = truncate_colormap(cm.get_cmap('gist_rainbow'), 0.05, 0.65)
# dark_colormap = truncate_colormap(cm.get_cmap('gist_rainbow_r'), 0.35, 0.95)


basic_color_dict = {
    'model': ['teal', 'lightseagreen',],
    'benchmark': ['coral', 'darkorange'], 
    'compare': ['steelblue', 'skyblue'],
    'annotate': 'red', # firebrick
    'particles': ['blue', 'cyan', 'red', 'orange', 'lightgrey'],
}

basic_style_dict = {
    'model': {'linestyle': 'solid'},
    'compare': {'linestyle': 'solid'},
    'annotate': {'linestyle': 'dotted', 'lw': 2.5}, # 2
    'histogram': ['solid', 'dashed', 'dotted',]
}


basic_pos_dict = {False: {False: [0.66, 0.82], True: [0.1, 0.82]}, True: {False: [0.73, 0.87], True: [0.73, 0.37]}}
