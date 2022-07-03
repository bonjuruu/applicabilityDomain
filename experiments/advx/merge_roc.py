import argparse
import os
from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import interpolate

from adad.plot import plot_roc_list
from adad.utils import create_dir, open_json

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_CURRENT = os.path.join(PATH_ROOT, 'experiments', 'advx')
PATH_JSON = os.path.join(PATH_CURRENT, 'metadata.json')
METADATA = open_json(PATH_JSON)
assert len(METADATA['datasets']) == len(METADATA['filenames']), 'Found an error in metadata.json file.'


def estimate_tpr(x, y, x_interp):
    """Interpolate ROC on an evenly spaced X values."""
    assert x.shape == y.shape
    # Fix start and finish points
    f = interpolate.interp1d(
        np.concatenate(([0], x, [1])),
        np.concatenate(([0], y, [1])))
    yy = f(x_interp)
    return yy


def parse_roc(df_fpr, df_tpr):
    """Interpolate FPRs and TPRs from multiple runs, and return a mean array"""
    assert df_fpr.shape == df_tpr.shape
    _x = np.linspace(0.01, 0.99, num=100 - 2)
    _y = []
    n_col = len(df_fpr.columns)
    for i in range(n_col):
        fpr = df_fpr.iloc[:, i].to_numpy()
        tpr = df_tpr.iloc[:, i].to_numpy()
        yy = estimate_tpr(fpr, tpr, _x)
        _y.append(yy)
    _y = np.array(_y)
    y_mean = _y.mean(axis=0)
    # Add start and finish points
    return np.concatenate(([0], _x, [1.])), np.concatenate(([0], y_mean, [1.]))


def merge_roc(path, indices):
    # Save mean ROC values into a folder called "mean"
    path_output = os.path.join(path, 'mean')
    create_dir(path_output)

    paths_roc = [os.path.join(path, f'run_{i}', 'roc') for i in indices]
    datasets = METADATA['datasets']
    attacks = METADATA['attacks']
    defenses = METADATA['ad']
    for dataname in datasets:
        for attack in attacks:
            for defense in defenses:
                # Merge results from multiple runs
                filenames = sorted(glob(os.path.join(paths_roc[0], f'roc_{defense}_{dataname}_{attack}_*.csv')))
                filenames = [os.path.basename(f) for f in filenames]
                for f in filenames:
                    df_roc = pd.DataFrame()
                    for i, path_csv in enumerate([os.path.join(p, f) for p in paths_roc]):
                        try: 
                            df_temp = pd.read_csv(path_csv)
                            df_roc[f'fpr_{i}'] = df_temp['fpr']
                            df_roc[f'tpr_{i}'] = df_temp['tpr']
                        except FileNotFoundError:
                            print(f'{path_csv} does not exist.', )
                    path_output_temp = os.path.join(path_output, 'roc_mean_{}.csv'.format(Path(f).stem[len('roc_'):]))
                    df_roc.to_csv(path_output_temp, index=False)
                    print('Save to:', path_output_temp)

                # Generate plot
                fprs = []
                tprs = []
                legends = []
                paths_mean_roc = [os.path.join(path_output, 'roc_mean_{}.csv'.format(Path(f).stem[len('roc_'):])) for f in filenames]
                xx = np.arange(0, 1.0, 0.01)
                for path_mean_roc in paths_mean_roc:
                    try:
                        df = pd.read_csv(path_mean_roc)
                        columns = df.columns.to_numpy()
                        columns_fpr = columns[[c[:len('fpr_')] == 'fpr_' for c in columns]]
                        columns_tpr = columns[[c[:len('tpr_')] == 'tpr_' for c in columns]]
                        fpr, tpr = parse_roc(df[columns_fpr], df[columns_tpr])
                        fprs.append(fpr)
                        tprs.append(tpr)
                        epsilon = Path(path_mean_roc).stem.split('_')[-1]
                        legends.append(epsilon)
                    except FileNotFoundError:
                        print(f'{path_mean_roc} does not exist.', )
                path_plot = os.path.join(path_output, f'roc_mean_{defense}_{dataname}_{attack}.pdf')
                plot_roc_list(
                    fprs,
                    tprs,
                    legend=[f'e={l}' for l in legends],
                    title=f'ROC for {dataname} on {attack.upper()}',
                    figsize=(6, 6),
                    path=path_plot,
                )
                print('Save plot to:', path_plot)


if __name__ == '__main__':
    """Examples
    python ./experiments/advx/merge_roc.py -p "./results/numeric/"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-i', '--indices', nargs='+', default=list(range(1, 6)))
    args = parser.parse_args()
    print('Received args:', args)

    path = str(Path(args.path).absolute())
    indices = args.indices

    print('Path:', path)
    print('Indices:', indices)

    merge_roc(path, indices)
