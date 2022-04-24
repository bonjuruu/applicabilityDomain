"""
Split data and then train a classifier.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from adad.models.numeric import NumericModel
from adad.torch_utils import evaluate, train_model, numpy_2_dataloader
from adad.utils import create_dir, open_csv, open_json, time2str, to_csv

PATH_ROOT = Path(os.getcwd()).absolute()
path_current = os.path.join(PATH_ROOT, 'experiments', 'advx')
path_json = os.path.join(path_current, 'metadata.json')
METADATA = open_json(path_json)
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

    dataloader_train = numpy_2_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    dataloader_test = numpy_2_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = NumericModel(
        n_features=n_features,
        n_hidden=hidden_layer,
        n_classes=n_classes,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    path_model = os.path.join(path_output, 'clf')
    create_dir(path_model)
    path_model = os.path.join(path_model, f'{dataname}.torch')

    if os.path.exists(path_model) and not restart:
        model.load_state_dict(torch.load(path_model, map_location=device))
    else:
        # Train the clean model
        time_start = time.perf_counter()
        train_model(model, dataloader_train, optimizer,
                    loss_fn, device, max_epochs)
        time_elapsed = time.perf_counter() - time_start
        print('Time taken: {}'.format(time2str(time_elapsed)))
        # Save model
        torch.save(model.state_dict(), path_model)

    ############################################################################
    # Step 3: Evaluate results
    acc_train, loss_train = evaluate(dataloader_train, model, loss_fn, device)
    acc_test, loss_test = evaluate(dataloader_test, model, loss_fn, device)
    print('Train acc: {:.2f} loss: {:.3f}'.format(acc_train * 100, loss_train))
    print(' Test acc: {:.2f} loss: {:.3f}'.format(acc_test * 100, loss_test))


if __name__ == '__main__':
    """Example:
    python ./experiments/advx/step2_train_clf.py -d abalone
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
