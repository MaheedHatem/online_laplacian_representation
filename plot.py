import numpy as np
from math import factorial
from matplotlib import pyplot as plt
import argparse
import yaml
from matplotlib.ticker import ScalarFormatter, MultipleLocator

from IPython.display import set_matplotlib_formats
from tol_colors import tol_cmap, tol_cset
import argparse

plt.rcParams['svg.fonttype'] = 'none'
set_matplotlib_formats('svg')

def compare_results(args):
    with open(f"src/Comparisons/{args.comparison_list}.yaml", "r") as f:
        data = yaml.safe_load(f)
    results_dirs = data['results_dirs']
    plt_title = data['plt_title']
    bar_chart = data.get("bar", False)
    x_label = data['x_label']
    y_label = data['y_label']
    column = data.get('column', 1)
    scores = {}
    ci = {}
    steps = []
    colors = tol_cset('high-contrast')
    
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 15})
    for i, (label, results_paths) in enumerate(results_dirs.items()):
        results = []
        for result_dir in results_paths:
            results.append(np.loadtxt(f"{args.parent_folder}/{result_dir}/results.csv", delimiter=','))
        results = np.stack(results, axis=2)
        if bar_chart:
            results = results[-20:]
            scores[label] = np.mean(results, axis=(0,2))
            ci[label] = 1.96 * np.std(results, axis=(0,2))/np.sqrt(results.shape[0]*results.shape[2])
        else:
            scores[label] = np.mean(results, axis=2)
            ci[label] = 1.96 * np.std(results, axis = 2)/np.sqrt(results.shape[2])
        
        if bar_chart:
            plt.bar(f"{label}", scores[label][column], yerr=ci[label][column], color=colors[0])
        else:
            indices = range(0,len(scores[label][:, 0]),2)
            plt.plot(scores[label][indices, 0], scores[label][indices,column], label=label, color=colors[i])
            plt.fill_between(scores[label][indices, 0], (scores[label][indices,column]-ci[label][indices,column]), (scores[label][indices,column]+ci[label][indices,column]), color=colors[i], alpha=.1)
    
    plt.tick_params(bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))  # Always show scientific notation

    plt.gca().xaxis.set_major_formatter(formatter)
    #plt.gca().xaxis.set_major_locator(MultipleLocator(50000))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.ylim(0, 1)
    #plt.title(plt_title)
    if not bar_chart:
        legend_loc = data.get('loc', 'best')
        if isinstance(legend_loc, list):
            legend_loc = tuple(legend_loc)
        plt.legend(loc=legend_loc, frameon=False)
    plt.savefig(f'src/Figures/SVG/{args.comparison_list}.svg')
    plt.savefig(f'src/Figures/{args.comparison_list}.pdf')
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_list")
    parser.add_argument("parent_folder", default=".")
    args = parser.parse_args()
    compare_results(args)