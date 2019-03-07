import matplotlib.pyplot as plt

from .math import confidence_interval


__all__ = ['plot_against_supervised', 'plot_statistics']


def plot_against_supervised(ss_methods, data_source, classifier, get_results_func, to_plot='class_acc', display=True, save_file=None):
    """
    Function to plot results of one test against fully supervised.
    Args:
        ss_methods (dict(str,dict)): maps ss_method_name to their values (algorithm, encoder, and similarity measure)
        data_source (str): which dataset we're considering
        classifier (str): type of classifier used
        to_plot (str): indicating what we're plotting (class_acc|p|r|f1)
    Returns:
        None
    """
    baseline_results = get_results_func('supervised', data_source, classifier)
    fracs = baseline_results['fracs']
    baseline_means = baseline_results[f'{to_plot}_means']
    baseline_cis = [confidence_interval(0.95, std, 10) for std in baseline_results[f'{to_plot}_stds']]
    plt.errorbar(fracs, baseline_means, yerr=baseline_cis, fmt='ro-', ecolor='red', elinewidth=0.8, capsize=1, label=f'fully supervised ({classifier})')

    for method_name, v in ss_methods.items():
        results = get_results_func(v['algorithm'], data_source, classifier, encoder=v['encoder'], similarity_measure=v['similarity'])
        means = [baseline_means[0]] + results[f'{to_plot}_means']
        cis = [baseline_cis[0]] + [confidence_interval(0.95, std, 10) for std in results[f'{to_plot}_stds']]
        plt.errorbar(fracs, means, yerr=cis, fmt=v['colour'], ecolor=v['ecolour'], elinewidth=0.8, capsize=1, label=f'{method_name}')

    plot_name = ' '.join(to_plot.split('_'))
    plt.title(f'{plot_name} on different fractions of labeled data.') # include data_source + classifier info in title ultimately.
    plt.ylabel(plot_name)
    plt.xticks([0.1*i for i in range(0,11)])
    plt.xlabel('fraction of labeled data')
    plt.legend()
    plt.grid(b=True)

    if save_file:
        plt.savefig(save_file)
    if display:
        plt.show()


def plot_statistics(ps, data_source, classifier, get_results_func, statistics=['class_acc'] ,display=True, save_file=None):
    results = get_results_func(ps['algorithm'], data_source, classifier, encoder=ps['encoder'], similarity_measure=ps['similarity'])
    fracs = results['fracs']
    for s, color in statistics:
        means = results[f'{s}_means']
        cis = [confidence_interval(0.95, std, 10) for std in results[f'{s}_stds']]
        plt.errorbar(fracs, means, yerr=cis, fmt=f'{color}o-', ecolor=color, elinewidth=0.8, capsize=1, label=f'{s}')
    
    plt.title(f'classification statistics on different fractions of labeled data.') # include data_source + classifier info in title ultimately.
    plt.ylabel("")
    plt.xticks([0.1*i for i in range(0,11)])
    plt.xlabel('fraction of labeled data')
    plt.legend()
    plt.grid(b=True)
    
    if save_file:
        plt.savefig(save_file)
    if display:
        plt.show()
