import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint

from adad.utils import create_dir, time2str

FINGER_PRINTS = ['ecfp', 'rdkit', 'maccs']
COL_NAME = 'smiles'


def gen_fp(input, output, method, column):
    start = time.process_time()
    df = pd.read_csv(input)
    smiles = df[column].to_list()
    results = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if method == 'ecfp':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        elif method == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:  # method == 'rdkit'
            fp = RDKFingerprint(mol)
        results.append(fp.ToList())
    time_elapse = time.process_time() - start
    print(f'Total run time: {time2str(time_elapse)}')
    results = np.array(results, dtype=int)
    assert results.shape[0] == df.shape[0]
    np.save(output, results, allow_pickle=True)


if __name__ == '__main__':
    """
    Example:
    python ./experiments/gen_fingerprints.py -f ./data/smiles/Ames_smiles.csv -m maccs -o ./data/maccs/Ames_maccs.npy
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The file path of the data.')
    parser.add_argument('-m', '--method', type=str, choices=FINGER_PRINTS,
                        required=True, help='Two methods are available: ecfp and rdkit.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output path.')
    parser.add_argument('-c', '--column', default=COL_NAME)
    args = parser.parse_args()
    filepath = Path(args.filepath).absolute()
    output = Path(args.output).absolute()
    method = args.method
    column = args.column

    print(f'  Path: {filepath}')
    print(f'Output: {output}')
    print(f'Method: {method}')
    print(f'Column: {column}')
    create_dir(Path(output).parent)

    gen_fp(filepath, output, method, column)
