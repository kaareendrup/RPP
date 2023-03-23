
import numpy as np
import pathlib
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


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


def get_rates(x, y, thresholds, target_rates, target_cuts, curve_type='ROC'):

    rates_list = []

    if target_rates is not None and target_cuts is not None:
        print('Both target cuts and rates specified. Using rates.')

    if target_rates is not None:
        for rate in target_rates:

            if curve_type=='ROC':
                diffs = abs(x - (1-rate))
            if curve_type=='PRC':
                diffs = abs(y - rate)
                
            idx = np.argmin(diffs)
            rates_list.append([x[idx], y[idx], thresholds[idx]])

    elif target_cuts is not None:
        for cut in target_cuts:
            diffs = abs(thresholds - cut)
            idx = np.argmin(diffs)
            rates_list.append([x[idx], y[idx], thresholds[idx]])

    return rates_list


def add_rates(axs, model, curve_type=None):

    curve_type = model._target_curve_type if curve_type is None else curve_type

    for (i_x, i_y, ithreshold) in model._performance_rates[curve_type]:
        axs[-1].scatter(i_x, i_y, color='k', s=10, zorder=3)
        
        if curve_type=='ROC':
            axs[-1].text(i_x*1.2, i_y-.005, 'Cut: %.4g'%ithreshold, va='top', fontsize=9)
            axs[-1].text(i_x*1.2, i_y-.035, 'FPR: %.2g'%(i_x*100)+'%', va='top', fontsize=9)
            axs[-1].text(i_x*1.2, i_y-.065, 'TPR: %.4g'%(i_y*100)+'%', va='top', fontsize=9)

        if curve_type=='PRC':
            axs[-1].text(i_x-.15, i_y-.02, 'Cut: %.3g'%ithreshold, va='top', fontsize=9)
            axs[-1].text(i_x-.15, i_y-.05, 'Precision: %.3g'%i_y, va='top', fontsize=9)
            axs[-1].text(i_x-.15, i_y-.08, 'Recall: %.3g'%i_x, va='top', fontsize=9)


curve_config_dict = {
    'ROC': [
        roc_curve, roc_auc_score, 'FPR', 'TPR'
    ],
    'PR': [
        PR_flip_outputs, average_precision_score, 'Efficiency (Recall)', 'Purity (Precision)'
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
