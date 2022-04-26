"""
Split data and then train a classifier.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from adad.torch_utils import NNClassifier
from adad.utils import create_dir, open_csv, open_json, time2str, to_csv

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_CURRENT = os.path.join(PATH_ROOT, 'experiments', 'advx')
PATH_JSON = os.path.join(PATH_CURRENT, 'metadata.json')
METADATA = open_json(PATH_JSON)
assert len(METADATA['datasets']) == len(METADATA['filenames']), 'Found an error in metadata.json file.'
PATH_DATA = os.path.join(PATH_ROOT, 'data', 'numeric', 'preprocessed')


def split_n_train(dataname, data_filename, filepath, path_output, testsize, path_params, restart):
    path_data = os.path.join(filepath, data_filename)
    X, y, cols = open_csv(path_data, label_name='Class')

    # Step 1: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testsize)
    # Same ratio is used for validation and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=X_test.shape[0])
    print(f'Train: {X_train.shape[0]} validation: {X_val.shape[0]} test: {X_test.shape[0]}')

    create_dir(os.path.join(path_output, 'train'))
    to_csv(X_train, y_train, cols, os.path.join(path_output, 'train', f'{dataname}_train.csv'))
    create_dir(os.path.join(path_output, 'validation'))
    to_csv(X_val, y_val, cols, os.path.join(path_output, 'validation', f'{dataname}_val.csv'))
    create_dir(os.path.join(path_output, 'test'))
    to_csv(X_test, y_test, cols, os.path.join(path_output, 'test', f'{dataname}_test.csv'))

    ############################################################################
    # Step 2: Train classifier
    n_features = X_train.shape[1]
    n_classes = np.unique(y_train).shape[0]
    print(f'# of features: {n_features}, # of labels: {n_classes}')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Running on CPU!')

    params = open_json(path_params)
    batch_size = params['batch_size']
    hidden_layer = params['hidden_layer']
    lr = params['learning_rate']
    max_epochs = params['max_epochs']
    momentum = params['momentum']
    print('# of neurons in hidden layer:', hidden_layer)

    clf = NNClassifier(
        input_dim=n_features,
        hidden_dim=hidden_layer,
        output_dim=n_classes,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=0.,
        device=device)

    # Train the model
    path_model = os.path.join(path_output, 'clf', dataname)
    time_start = time.perf_counter()
    clf.fit(X_train, y_train)
    time_elapsed = time.perf_counter() - time_start
    print('Time taken: {}'.format(time2str(time_elapsed)))
    # Save model
    # torch.save(model.state_dict(), path_model)
    clf.save(path_model)

    ############################################################################
    # Step 3: Evaluate results
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    print('Train acc: {:.2f}%'.format(acc_train * 100))
    print(' Test acc: {:.2f}%'.format(acc_test * 100))


if __name__ == '__main__':
    """Example:
    python ./experiments/advx/step2_train_clf.py -d abalone
    python ./experiments/advx/step2_train_clf.py -d abalone -r 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-f', '--filepath', type=str, default=PATH_DATA)
    parser.add_argument('-o', '--output', type=str, default='results/numeric')
    parser.add_argument('--testsize', type=float, default=0.2)
    parser.add_argument('-p', '--params', type=str, default='./experiments/advx/params.json')
    parser.add_argument('-r', '--restart', type=int, default=1)
    parser.add_argument('-i', '--index', type=int, default=1)
    args = parser.parse_args()
    print('Received args:', args)

    dataname = args.data
    filepath = str(Path(args.filepath).absolute())
    path_output = str(Path(args.output).absolute())
    testsize = args.testsize
    path_params = str(Path(args.params).absolute())
    restart = True if args.restart == 1 else False
    index = args.index

    path_output = os.path.join(path_output, f'run_{index}')
    idx_data = METADATA['datasets'].index(dataname)
    data_filename = METADATA['filenames'][idx_data]
    print('Dataset:', dataname)
    print('ROOT directory:', PATH_ROOT)
    print('Save to:', path_output)
    split_n_train(dataname, data_filename, filepath, path_output, testsize, path_params, restart)
