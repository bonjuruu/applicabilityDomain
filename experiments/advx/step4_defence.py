import argparse
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, roc_curve


from adad.bounding_box import PCABoundingBox
from adad.distance import DAIndexDelta, DAIndexGamma, DAIndexKappa
from adad.probability import ProbabilityClassifier
from adad.torch_utils import NNClassifier, get_correct_examples
from adad.utils import open_csv, open_json

PATH_ROOT = Path(os.getcwd()).absolute()
path_current = os.path.join(PATH_ROOT, 'experiments', 'advx')
path_json = os.path.join(path_current, 'metadata.json')
METADATA = open_json(path_json)
assert len(METADATA['datasets']) == len(METADATA['filenames']), 'Found an error in metadata.json file.'
PATH_DATA = os.path.join(PATH_ROOT, 'data', 'numeric', 'preprocessed')


def get_ad(adname):
    if adname == 'gamma':
        ad = DAIndexGamma
    elif adname == 'kappa':
        ad = DAIndexKappa
    elif adname == 'delta':
        ad = DAIndexDelta
    elif adname == 'boundingbox':
        ad = PCABoundingBox
    elif adname == 'prob':
        ad = ProbabilityClassifier
    else:
        raise NotImplementedError
    return ad


def run_ad(dataname, path_output, att_name, ad_name):
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
    path_model = os.path.join(path_output, 'clf', dataname)
    if not os.path.exists(path_model):
        raise FileExistsError
    clf = NNClassifier()
    clf.load(path_model)

    # Step 3: Filter data
    X_train_fil, y_train_fil = get_correct_examples(clf.clf, X_train, y_train, clf.device)
    X_val_fil, y_val_fil = get_correct_examples(clf.clf, X_val, y_val, clf.device)
    X_test_fil, y_test_fil = get_correct_examples(clf.clf, X_test, y_test, clf.device)

    print('After filtering:')
    print('{:12s} Before: {:6d} After: {:6d}'.format('Train', y_train.shape[0], len(y_train_fil)))
    print('{:12s} Before: {:6d} After: {:6d}'.format('Validation', y_val.shape[0], len(y_val_fil)))
    print('{:12s} Before: {:6d} After: {:6d}'.format('Test', y_test.shape[0], len(y_test_fil)))

    # Evaluate clean data on pretrained classifier after filtering
    acc_train = clf.score(X_train_fil, y_train_fil)
    acc_val = clf.score(X_val_fil, y_val_fil)
    acc_test = clf.score(X_test_fil, y_test_fil)
    print('{:12s} acc: {:6.2f}%'.format('Train', acc_train * 100))
    print('{:12s} acc: {:6.2f}%'.format('Validation', acc_val * 100))
    print('{:12s} acc: {:6.2f}%'.format('Test', acc_test * 100))

    files_advx = sorted(glob(os.path.join(path_output, att_name, f'{dataname}*')))
    print('# of advx sets:', len(files_advx))
    # for file_advx in files_advx:
    #     print('Current file:', file_advx)
    #     X_advx, y_true, _ = open_csv(file_advx)
    #     dataloader_advx_fil = numpy_2_dataloader(X_advx, y_true, batch_size=batch_size, shuffle=False)
    #     acc_advx, _ = evaluate(dataloader_advx_fil, model, loss_fn, device)
    #     print('Advx acc: {:.2f}%'.format(acc_advx * 100))

    file_advx = files_advx[0]
    print('Current file:', file_advx)
    X_advx, y_true, _ = open_csv(file_advx)
    acc_advx = clf.score(X_advx, y_true)
    print('{:12s} acc: {:6.2f}%'.format('Advx', acc_advx * 100))

    # Merge clean and adversarial samples
    assert X_advx.shape == X_test_fil.shape
    X_full = np.vstack((X_test_fil, X_advx))
    lbl_full = np.concatenate((np.zeros(X_test_fil.shape[0]), np.ones(X_advx.shape[0])))

    # Step 4: Run Applicability Domain
    ApplicabilityDomain = get_ad(ad_name)
    ad_params = METADATA['adParams'][ad_name]
    ad_params['clf'] = clf
    ad = ApplicabilityDomain(**ad_params)
    ad.fit(X_train_fil, y_train_fil)

    dist_measure = ad.measure(X_full)

    # TODO: code below is for debugging only!
    fpr, tpr, _ = roc_curve(lbl_full, dist_measure)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=ad_name)
    display.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, f'{dataname}_{ad_name}.pdf'), dpi=300)
    


if __name__ == '__main__':
    """Examples:
    python ./experiments/advx/step4_defence.py -d abalone -o "./results/numeric/run_1/" -a fgsm --ad gamma
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, choices=METADATA['datasets'])
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-a', '--attack', type=str, required=True, choices=METADATA['attacks'])
    parser.add_argument('--ad', type=str, required=True, choices=METADATA['ad'],
                        help='Applicability Domain name.')
    args = parser.parse_args()
    print('Received args:', args)

    dataname = args.data
    path_output = str(Path(args.output).absolute())
    att_name = args.attack
    ad_name = args.ad

    print('Dataset:', dataname)
    print('Attack:', att_name)
    print('AD:', ad_name)
    print('Path:', path_output)

    run_ad(dataname, path_output, att_name, ad_name)
