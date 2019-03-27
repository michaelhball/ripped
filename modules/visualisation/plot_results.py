import matplotlib.pyplot as plt

from modules.utilities.math import confidence_interval

__all__ = ['plot_one_statistic']


def plot_one_statistic(methods, data_source, classifier, get_results, plot_type='embeddings', to_plot='f1', title='DOOT'):
    """
    Function to plot results of one test against fully supervised.
    Args:
        methods (dict(str,dict)): maps ss_method_name to their values (algorithm, encoder, and similarity measure)
        data_source (str): which dataset we're considering
        classifier (str): type of classifier used
        get_results (func): function to get results from relevant results file
        plot_type (str): embeddings|algos (either comparing different embeddings or different SS methods for one embedding type)
        to_plot (str): statistic we're plotting (class_acc|p|r|f1)
        title (str): chart title
    Returns:
        None
    """
    with plt.style.context('seaborn-whitegrid'):
        # plot baseline
        bline_res = get_results('supervised', data_source, classifier)
        fracs = bline_res['fracs']
        bline_means = bline_res[f'{to_plot}_means']
        bline_cis = [confidence_interval(0.95, std, bline_res['n']) for std in bline_res[f'{to_plot}_stds']]
        if plot_type == "embeddings":
            plt.errorbar(fracs, bline_means, yerr=bline_cis, fmt='C0.-', ecolor='C0', elinewidth=0.8, capsize=1, label=f'fully-supervised')

        # plot all other methods
        for idx, (name, v) in enumerate(methods.items()):
            res = get_results(v['encoder'], data_source, classifier, algorithm=v['algorithm'])
            means = [bline_means[0]] + res[f'{to_plot}_means']
            cis = [bline_cis[0]] + [confidence_interval(0.95, std, res['n']) for std in res[f'{to_plot}_stds']]
            plt.errorbar(fracs, means, yerr=cis, fmt=f'C{idx+1}.-', ecolor=f'C{idx+1}', elinewidth=0.8, capsize=1, label=f'{name}')

        plt.title(title)
        plt.ylabel(to_plot)
        plt.xlabel('fraction of labeled data')
        plt.legend()
        plt.grid(b=True)
        plt.show()
