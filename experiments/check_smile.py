import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from adad.utils import create_dir, to_json
from rdkit import Chem

COL_NAME = 'smiles'


def check_smile(input, output, column):
    df = pd.read_csv(input)
    smiles = df[column].to_list()
    idx_na = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            idx_na.append(i)
    print(f'# of smiles cannot convert: {len(idx_na)}')
    df = df.drop(index=idx_na)
    filename = os.path.basename(input)
    filename = os.path.splitext(filename)[0]
    df.to_csv(os.path.join(output, filename + '.csv'), index=False)
    data = {'indices': np.array(idx_na, dtype=int)}
    print(data)
    to_json(data, os.path.join(output, filename + '.json'))


if __name__ == '__main__':
    """
    Example:
    python ./experiments/check_smile.py -f ./data/smiles/Ames_smiles.csv -o ./data2/smiles
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The file path of the data.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output path.')
    parser.add_argument('-c', '--column', default=COL_NAME)
    args = parser.parse_args()
    filepath = Path(args.filepath).absolute()
    output = Path(args.output).absolute()
    column = args.column

    print(f'  Path: {filepath}')
    print(f'Output: {output}')
    print(f'Column: {column}')
    create_dir(Path(output))

    check_smile(filepath, output, column)
