import argparse
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem

from adad.utils import create_dir, time2str

COL_NAME = 'smiles'


def check_smile(input, output, column):
    start = time.process_time()
    df = pd.read_csv(input)
    smiles = df[column].to_list()
    idx_na = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            idx_na.append(i)
    print(f'# of smiles cannot convert: {len(idx_na)}')
    df = df.drop(index=idx_na)
    df.to_csv(output, index=False)
    time_elapse = time.process_time() - start
    print(f'Total run time: {time2str(time_elapse)}')


if __name__ == '__main__':
    """
    Example:
    python ./experiments/check_smile.py -f ./data/smiles/Ames_smiles.csv -o ./data2/smiles/ames_smiles.csv
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
    create_dir(Path(output).parent)

    check_smile(filepath, output, column)
