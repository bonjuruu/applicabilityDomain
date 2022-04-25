"""
Train adversarial examples
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from adad.models.advx_attack import AdvxAttack
from adad.models.torch_train import get_correct_examples
from adad.torch_utils import NNClassifier
from adad.utils import create_dir, open_csv, open_json, time2str, to_csv

PATH_ROOT = Path(os.getcwd()).absolute()
PATH_CURRENT = os.path.join(PATH_ROOT, 'experiments', 'advx')
PATH_JSON = os.path.join(PATH_CURRENT, 'metadata.json')
METADATA = open_json(PATH_JSON)
PATH_DATA = os.path.join(PATH_ROOT, 'data', 'numeric', 'preprocessed')

assert len(METADATA['datasets']) == len(METADATA['filenames']), 'Found an error in metadata.json file.'


def train_attacks(dataname, path_output, attack, epsilons):
    path_advx = os.path.join(path_output, attack)
    create_dir(path_advx)

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
    clip_values = (X_train.min(), X_train.max())
    print('Clip values:', clip_values)

    # Step 2: Load model
    path_model = os.path.join(path_output, 'clf', dataname)
    if not os.path.exists(path_model):
        raise FileExistsError
    clf = NNClassifier()
    clf.load(path_model)

    # Evaluate clean data on pretrained classifier
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    print('Train acc: {:.2f}%'.format(acc_train * 100))
    print(' Test acc: {:.2f}%'.format(acc_test * 100))

    # Step 3: Filter data
    X_train_fil, y_train_fil = get_correct_examples(clf.clf, X_train, y_train, clf.device)
    X_val_fil, y_val_fil = get_correct_examples(clf.clf, X_val, y_val, clf.device)
    X_test_fil, y_test_fil = get_correct_examples(clf.clf, X_test, y_test, clf.device)

    print('After filtering:')
    print(f'     Train before filtering: {y_train.shape[0]} after: {len(X_train_fil)}')
    print(f'Validation before filtering: {y_val.shape[0]} after: {len(X_val_fil)}')
    print(f'      Test before filtering: {y_test.shape[0]} after: {len(X_test_fil)}')

    # Evaluate clean data on pretrained classifier after filtering
    acc_train = clf.score(X_train_fil, y_train_fil)
    acc_val = clf.score(X_val_fil, y_val_fil)
    acc_test = clf.score(X_test_fil, y_test_fil)
    print('     Train acc: {:.2f}%'.format(acc_train * 100))
    print('Validation acc: {:.2f}%'.format(acc_val * 100))
    print('      Test acc: {:.2f}%'.format(acc_test * 100))

    # Step 4: Preform attack
    attack_model = AdvxAttack(
        clf.clf,
        loss_fn=clf.loss_fn,
        optimizer=clf.optimizer,
        n_features=clf.input_dim,
        n_classes=clf.output_dim,
        att_name=attack,
        device=clf.device,
        clip_values=clip_values,
        batch_size=clf.batch_size)

    for e in epsilons:
        time_start = time.perf_counter()
        adv_fil = attack_model.generate(X_test_fil, e)
        time_elapsed = time.perf_counter() - time_start
        print(f'Generating adversarial examples at {e:.3} took {time2str(time_elapsed)}')

        acc_adv = clf.score(adv_fil, y_test_fil)
        print('Advx (e={:.3f}) acc: {:.2f}%'.format(e, acc_adv * 100))

        # Step 5: save results
        path_advx_temp = os.path.join(path_advx, f'{dataname}_{attack}_{np.round(e, 3):.3f}.csv')
        to_csv(adv_fil, y_test_fil, cols, path_advx_temp)


if __name__ == '__main__':
    """Example:
    python ./experiments/advx/step3_attack.py -d abalone -a fgsm -e 0.3 1.0
    python ./experiments/advx/step3_attack.py -d abalone -a cw2 -e 10
    python ./experiments/advx/step3_attack.py -d abalone -a apgd -e 0.3 1.0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-o', '--output', type=str, default='results/numeric')
    parser.add_argument('-i', '--index', type=int, default=1)
    parser.add_argument('-a', '--attack', type=str, required=True, choices=METADATA['attacks'])
    parser.add_argument('-e', '--epsilons', nargs='+', default=[0.3])  # default value is for testing only!
    args = parser.parse_args()
    print('Received args:', args)

    dataname = args.data
    path_output = str(Path(args.output).absolute())
    index = args.index
    attack = args.attack
    epsilons = np.array([float(e) for e in args.epsilons])

    path_output = os.path.join(path_output, f'run_{index}')
    idx_data = METADATA['datasets'].index(dataname)
    data_filename = METADATA['filenames'][idx_data]
    print('Dataset:', dataname)
    print('Attack:', attack)
    print('ROOT directory:', PATH_ROOT)
    print('Save to:', path_output)

    train_attacks(dataname, path_output, attack, epsilons)
