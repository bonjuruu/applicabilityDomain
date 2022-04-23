"""
Train adversarial examples
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adad.models.numeric import NumericModel
from adad.models.torch_train import evaluate, get_correct_examples
from adad.utils import create_dir, open_csv, open_json, time2str, to_csv

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_CURRENT = os.path.join(PATH_ROOT, 'experiments', 'advx')
PATH_JSON = os.path.join(PATH_CURRENT, 'metadata.json')
METADATA = open_json(PATH_JSON)
PATH_DATA = os.path.join(PATH_ROOT, 'data', 'numeric', 'preprocessed')

assert len(METADATA['datasets']) == len(METADATA['filenames']), 'Found an error in metadata.json file.'


def train_attacks(dataname, data_filename, path_output, path_params, attack, epsilon):
    # Step 1: Load test set
    path_train = os.path.join(path_output, 'train', f'{dataname}_train.csv')
    path_val = os.path.join(path_output, 'validation', f'{dataname}_val.csv')
    path_test = os.path.join(path_output, 'test', f'{dataname}_test.csv')
    for p in [path_train, path_val, path_test]:
        if not os.path.exists(p):
            raise FileExistsError
    X_train, y_train, cols = open_csv(path_train)
    X_val, y_val, cols = open_csv(path_val)
    X_test, y_test, cols = open_csv(path_test)

    # Step 2: Load model
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

    model = NumericModel(
        n_features=n_features,
        n_hidden=hidden_layer,
        n_classes=n_classes,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    path_model = os.path.join(path_output, 'clf', f'{dataname}.torch')
    if not os.path.exists(path_model):
        raise FileExistsError
    model.load_state_dict(torch.load(path_model, map_location=device))

    # Create instances for PyTorch dataloaders
    dataset_train = TensorDataset(torch.from_numpy(X_train).type(torch.float32),
                                  torch.from_numpy(y_train).type(torch.int64))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = TensorDataset(torch.from_numpy(X_val).type(torch.float32),
                                torch.from_numpy(y_val).type(torch.int64))
    # dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataset_test = TensorDataset(torch.from_numpy(X_test).type(torch.float32),
                                 torch.from_numpy(y_test).type(torch.int64))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Evaluate clean data on pretrained classifier
    acc_train, loss_train = evaluate(dataloader_train, model, loss_fn, device)
    acc_test, loss_test = evaluate(dataloader_test, model, loss_fn, device)
    print('Train acc: {:.2f} loss: {:.3f}'.format(acc_train * 100, loss_train))
    print(' Test acc: {:.2f} loss: {:.3f}'.format(acc_test * 100, loss_test))

    # Step 3: Filter data
    # Return tensor dataset
    dataset_train_fil = get_correct_examples(
        model, dataset_train, X_train.shape, device, return_tensor=False)
    dataset_val_fil = get_correct_examples(
        model, dataset_val, X_val.shape, device, return_tensor=False)
    dataset_test_fil = get_correct_examples(
        model, dataset_test, X_test.shape, device, return_tensor=False)

    print('After filtering:')
    print(f'     Train before filtering: {y_train.shape[0]} after: {len(dataset_train_fil)}')
    print(f'Validation before filtering: {y_val.shape[0]} after: {len(dataset_val_fil)}')
    print(f'      Test before filtering: {y_test.shape[0]} after: {len(dataset_test_fil)}')

    dataloader_train_fil = DataLoader(dataset_train_fil, batch_size=batch_size, shuffle=False)
    dataloader_val_fil = DataLoader(dataset_val_fil, batch_size=batch_size, shuffle=False)
    dataloader_test_fil = DataLoader(dataset_test_fil, batch_size=batch_size, shuffle=False)

    # Evaluate clean data on pretrained classifier after filtering
    acc_train, loss_train = evaluate(dataloader_train_fil, model, loss_fn, device)
    acc_val, loss_val = evaluate(dataloader_val_fil, model, loss_fn, device)
    acc_test, loss_test = evaluate(dataloader_test_fil, model, loss_fn, device)
    print('     Train acc: {:.2f} loss: {:.3f}'.format(acc_train * 100, loss_train))
    print('Validation acc: {:.2f} loss: {:.3f}'.format(acc_val * 100, loss_val))
    print('      Test acc: {:.2f} loss: {:.3f}'.format(acc_test * 100, loss_test))

    # Step 4: Preform attack

    # Step 5: save results


if __name__ == '__main__':
    """Example:
    python ./experiments/advx/step3_attack.py -d abalone -a fgsm
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-o', '--output', type=str, default='results/numeric')
    parser.add_argument('-p', '--params', type=str, default='./experiments/advx/params.json')
    parser.add_argument('-i', '--index', type=int, default=1)
    parser.add_argument('-a', '--attack', type=str, required=True, choices=METADATA['attacks'])
    parser.add_argument('-e', '--epsilon', nargs='+', default=[0.3])  # default value is for testing only!
    args = parser.parse_args()
    dataname = args.data
    path_output = str(Path(args.output).absolute())
    path_params = str(Path(args.params).absolute())
    index = args.index
    attack = args.attack
    epsilon = args.epsilon

    path_output = os.path.join(path_output, f'run_{index}')
    idx_data = METADATA['datasets'].index(dataname)
    data_filename = METADATA['filenames'][idx_data]
    print('Dataset:', dataname)
    print('ROOT directory:', PATH_ROOT)
    print('Save to:', path_output)
    train_attacks(dataname, data_filename, path_output, path_params, attack, epsilon)
