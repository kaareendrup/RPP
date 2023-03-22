
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import pathlib
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


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


def make_plot_dir(name, target, dir):

    plot_dir = dir + '/plots/' + target + '_' + name + '/'

    path = pathlib.Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)

    return plot_dir


fiTQun_dict = {
    'v_u': 'fqmu_nll',
    'v_e': 'fqe_nll',
}


def PR_flip_outputs(y_true, probas_pred, pos_label=None):
    pre, rec, thresholds = precision_recall_curve(y_true, probas_pred, pos_label=pos_label)
    return rec, pre, thresholds


def ROC_get_rates(fpr, tpr, thresholds, target_rates, target_cuts):

    rates_list = []

    if target_rates is not None and target_cuts is not None:
        print('Both target cuts and rates specified. Using rates.')

    if target_rates is not None:
        for rate in target_rates:
            diffs = abs(fpr - (1-rate))
            idx = np.argmin(diffs)
            rates_list.append([fpr[idx], tpr[idx], thresholds[idx]])

    elif target_cuts is not None:
        for cut in target_cuts:
            diffs = abs(thresholds - cut)
            idx = np.argmin(diffs)
            rates_list.append([fpr[idx], tpr[idx], thresholds[idx]])

    return rates_list


def PR_get_rates(rec, pre, thresholds, target_rates, target_cuts):

    rates_list = []

    if target_rates is not None and target_cuts is not None:
        print('Both target cuts and rates specified. Using rates.')

    if target_rates is not None:
        for rate in target_rates:
            diffs = abs(pre - rate)
            idx = np.argmin(diffs)
            rates_list.append([rec[idx], pre[idx], thresholds[idx]])

    if target_cuts is not None:
        for cut in target_cuts:
            diffs = abs(thresholds - cut)
            idx = np.argmin(diffs)
            rates_list.append([rec[idx], pre[idx], thresholds[idx]])

    return rates_list


def ROC_add_rates(axs, model):

    for (ifpr, itpr, ithreshold) in model._performance_rates['ROC']:
        axs[-1].scatter(ifpr, itpr, color='k', s=10, zorder=3)
        axs[-1].text(ifpr*1.2, itpr-.005, 'Cut: %.4g'%ithreshold, va='top', fontsize=9)
        axs[-1].text(ifpr*1.2, itpr-.035, 'FPR: %.2g'%(ifpr*100)+'%', va='top', fontsize=9)
        axs[-1].text(ifpr*1.2, itpr-.065, 'TPR: %.4g'%(itpr*100)+'%', va='top', fontsize=9)


def PRC_add_rates(axs, model):

    for (irec, ipre, ithreshold) in model._performance_rates['PRC']:
        axs[-1].scatter(irec, ipre, color='k', s=10, zorder=3)
        axs[-1].text(irec-.15, ipre-.02, 'Cut: %.3g'%ithreshold, va='top', fontsize=9)
        axs[-1].text(irec-.15, ipre-.05, 'Precision: %.3g'%ipre, va='top', fontsize=9)
        axs[-1].text(irec-.15, ipre-.08, 'Recall: %.3g'%irec, va='top', fontsize=9)


curve_config_dict = {
    'ROC': [
        roc_curve, roc_auc_score, 'FPR', 'TPR', ROC_get_rates, ROC_add_rates
    ],
    'PR': [
        PR_flip_outputs, average_precision_score, 'Efficiency (Recall)', 'Purity (Precision)', PR_get_rates, ROC_add_rates
    ]
}


def target_extractor(target):

    if target is not None:
        if len(target) == 1:
            target, bg = target[0], target[0]
        elif len(target) == 2:
            target, bg = target[0], target[1]
        else:
            raise SystemExit('Please specify target rates and cuts as a list of lists.')
    else:
        bg = None

    return target, bg
    

def calculate_alpha(data):

    l = len(data)+1
    alpha = min(1, l**(-0.9)*1000)

    return alpha


def beautify_label(label):

    l = list(label)
    for i, c in enumerate(l):
        if c == 'v':
            if i == 0:
                if not l[i+1].isalnum():
                    l[i] = r'\nu'
            elif i == len(l)-1:
                if not l[i-1].isalnum():
                    l[i] = r'\nu'
            else:
                if not l[i-1].isalnum() and not l[i+1].isalnum():
                    l[i] = r'\nu'
                    
    label = "".join(l)
    return r'$%s$' % (label)
