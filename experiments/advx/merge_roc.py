import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from adad.utils import create_dir, open_json

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_CURRENT = os.path.join(PATH_ROOT, 'experiments', 'advx')
PATH_JSON = os.path.join(PATH_CURRENT, 'metadata.json')
METADATA = open_json(PATH_JSON)
assert len(METADATA['datasets']) == len(METADATA['filenames']), 'Found an error in metadata.json file.'


def merge_roc(path, indices):
    path_output = os.path.join(path, 'mean')
    create_dir(path_output)

    paths_roc = [os.path.join(path, f'run_{i}', 'roc') for i in indices]
    datasets = METADATA['datasets']
    attacks = METADATA['attacks']
    defenses = METADATA['ad']
    for dataname in datasets:
        for attack in attacks:
            for defense in defenses:
                # Get file names
                filenames = sorted(glob(os.path.join(paths_roc[0], f'roc_{defense}_{dataname}_{attack}_*.csv')))
                filenames = [os.path.basename(f) for f in filenames]
                for f in filenames:
                    df_roc = pd.DataFrame()
                    for i, path_csv in enumerate([os.path.join(p, f) for p in paths_roc]):
                        df_temp = pd.read_csv(path_csv)
                        df_roc[f'fpr_{i}'] = df_temp['fpr']
                        df_roc[f'tpr_{i}'] = df_temp['tpr']
                    path_output_temp = os.path.join(path_output, f'roc_mean_{Path(f).stem[4:]}.csv')
                    df_roc.to_csv(path_output_temp, index=False)
                    print('Save to:', path_output_temp)




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
