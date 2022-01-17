import os
from pathlib import Path
import pandas as pd
import numpy as np
from adad.utils import create_dir
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc

#Global variables
PATH_ROOT = os.getcwd()
CLASSIFIERS = ['KNeighborsClassifier', 'NNClassifier', 'RandomForestClassifier', 'SVC']
AD = ['DAIndexDelta', 'DAIndexGamma', 'DAIndexKappa', 'Magnet', 'PCABoundingBox', 'ProbabilityClassifier', 'SklearnFeatureSqueezing', 'SklearnRegionBasedClassifier']
DATASETS = ['Ames', 'BBBP', 'Cancer', 'CYP1A2', 'hERG', 'HIV', 'Liver']
GRAPH_FILE = ['CumulativeAccuracy.csv', 'PredictivenessCurves.csv', 'roc.csv']
COLOURS = ['#332288', '#88CCEE', '#44AA99', '#117733', '#DDCC77', '#CC6677', '#882255', '#AA4499']

parent_directory = os.path.join(PATH_ROOT, 'results')
save_directory = os.path.join(PATH_ROOT, 'plot_results')
graph_path = os.path.join(save_directory, "plot")
create_dir(graph_path)

def find_mean_dataset(dataset, experiment_dir, df_ca, df_pc, df_roc, name=None):
    """Finds the mean rate of all cv folds of a dataset. This is done for all types of graphs."""
    #CumulativeAccuracy
    
    #Find ca csv file location
    file = os.path.join(experiment_dir, f"{dataset}_{GRAPH_FILE[0]}")
    ca_csv = pd.read_csv(file, dtype=float)
    mean_x = np.linspace(0, 1, 100)
    
    #Add name to columns name if wanted
    if name != None:
        column = f"{dataset}_{name}"
    else:
        column = dataset

    #Find acc mean to 100 data points
    y = []
    for i in range(5):
        fold_acc = ca_csv[f'cv{i+1}_acc'].dropna().to_numpy()
        fold_rate = ca_csv[f'cv{i+1}_rate'].dropna().to_numpy()
        y.append(np.interp(mean_x, fold_rate, fold_acc))
    
    mean_y = pd.DataFrame(np.mean(y, axis=0), columns=[column])
    df_ca = pd.concat((df_ca, mean_y), axis=1)
    
    #AUC ROC
    #Find roc csv file location
    file = os.path.join(experiment_dir, f"{dataset}_{GRAPH_FILE[2]}")
    roc_csv = pd.read_csv(file, dtype=float)
    mean_fpr = np.linspace(0, 1, 100)
    
    #Find mean tpr as 100 data points
    tpr = []
    for i in range(5):
        fold_fpr = roc_csv[f'cv{i+1}_fpr'].dropna().to_numpy()
        fold_tpr = roc_csv[f'cv{i+1}_tpr'].dropna().to_numpy()
        interp_tpr = np.interp(mean_fpr, fold_fpr, fold_tpr)
        interp_tpr[0] = 0.0
        tpr.append(interp_tpr)
    
    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_tpr = pd.DataFrame(mean_tpr, columns=[column])
    df_roc = pd.concat((df_roc, mean_tpr), axis=1)
    
    #PredictivenessCurves
    #Find pc csv location
    file = os.path.join(experiment_dir, f"{dataset}_{GRAPH_FILE[1]}")
    pc_csv = pd.read_csv(file, dtype=float)
    mean_percentile = np.linspace(0, 1, 100)
    
    #Find mean of error rate as 100 data points
    err_rate = []
    for i in range(5):
        fold_percentile = pc_csv[f'cv{i+1}_percentile'].dropna().to_numpy()
        fold_err_rate = pc_csv[f'cv{i+1}_err_rate'].dropna().to_numpy()
        err_rate.append(np.interp(mean_percentile, fold_percentile, fold_err_rate))
        
    mean_err_rate = pd.DataFrame(np.mean(err_rate, axis=0), columns=[column])
    df_pc = pd.concat((df_pc, mean_err_rate), axis=1)
    
    return df_ca, df_pc, df_roc

def create_ca_graph(a_dict, path, name, fontsize=12, figsize=(8,8)):
    """Plot Cumulative Accuracy"""
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    mean_x = np.linspace(0, 1, 100)

    #Create line for each applicability domain
    for ad in AD:
        df_ca = a_dict[ad]['CumulativeAccuracy']
        y = []
        for dataset in df_ca.columns:
            y.append(df_ca[dataset].to_numpy())
        mean_y = np.mean(y, axis=0)
        ax.plot(mean_x, mean_y, alpha=1, color=COLOURS[AD.index(ad)], lw=2.5, label=ad)

    ax.set(xlim=[-0.01, 1.01], ylim=[0.5, 1.01])
    ax.legend(loc="lower right")
    ax.set_xlabel('Cumulative Rate')
    ax.set_ylabel('Cumulative Accuracy (%)')
    ax.set_title(f'{name} - Cumulative Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{name}_Cumulative_Accuracy.pdf'), dpi=300)
    
def create_pc_graph(a_dict, path, name, fontsize=12, figsize=(8,8)):
    """Plot Predictiveness Curves"""
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    mean_x = np.linspace(0, 1, 100)

    #Create line for each applicability domain
    for ad in AD:
        df_pc = a_dict[ad]['PredictivenessCurves']
        y = []
        for dataset in df_pc.columns:
            y.append(df_pc[dataset].to_numpy())
        mean_y = np.mean(y, axis=0)
        ax.plot(mean_x, mean_y, alpha=1, color=COLOURS[AD.index(ad)], lw=2.5, label=ad)

    ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 0.6])
    ax.legend(loc="upper left")
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Error Rate')
    ax.set_title(f'{name} - Predictiveness Curves (PC)')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{name}_Predictiveness_Curves.pdf'), dpi=300)
    
def create_roc_graph(a_dict, path, name, fontsize=12, figsize=(8,8)):
    """Plot ROC Curve"""
    plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots(figsize=figsize)
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    #Create line for each applicability domain
    for ad in AD:
        df_roc = a_dict[ad]['ROC']
        y = []
        for dataset in df_roc.columns:
            y.append(df_roc[dataset].to_numpy())
        mean_tpr = np.mean(y, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        display = RocCurveDisplay(fpr=mean_fpr, tpr=mean_tpr, roc_auc=mean_auc)
        display.plot(ax=ax, alpha=1, lw=2.5, color=COLOURS[AD.index(ad)], label=f"{ad} (AUC={mean_auc:.2f})")

    ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax.legend(loc="lower right")
    ax.set_title(f'{name} - ROC Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{name}_ROC.pdf'), dpi=300)
    
def run_classifier_graphs():
    """Create graph for each Classifier"""
    knn_dict = dict()
    nn_dict = dict()
    rf_dict = dict()
    svc_dict = dict()
    
    for classifier in CLASSIFIERS:
        save_path = os.path.join(save_directory, "classifier", classifier)
        create_dir(save_path)
        for method in AD:
            experiment_dir = os.path.join(parent_directory, f"{classifier}_{method}")
            df_ca = pd.DataFrame()
            df_pc = pd.DataFrame()
            df_roc = pd.DataFrame()
            for dataset in DATASETS:
                df_ca, df_pc, df_roc = find_mean_dataset(dataset, experiment_dir, df_ca, df_pc, df_roc)
            
            df_ca.to_csv(os.path.join(save_path, f'{method}_ca.csv'), index=False)
            df_pc.to_csv(os.path.join(save_path, f'{method}_pc.csv'), index=False)
            df_roc.to_csv(os.path.join(save_path, f'{method}_roc.csv'), index=False)
            
            if classifier == 'KNeighborsClassifier':
                knn_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            elif classifier == 'NNClassifier':
                nn_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            elif classifier == 'RandomForestClassifier':
                rf_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            else:
                svc_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
    
    classifier_path = os.path.join(graph_path, 'classifier')
    create_dir(classifier_path)
    
    create_ca_graph(knn_dict, classifier_path, 'KNeighborsClassifier')
    create_ca_graph(nn_dict, classifier_path, 'NNClassifier')
    create_ca_graph(rf_dict, classifier_path, 'RandomForestClassifier')
    create_ca_graph(svc_dict, classifier_path, 'SVC')
    
    create_pc_graph(knn_dict, classifier_path, 'KNeighborsClassifier')
    create_pc_graph(nn_dict, classifier_path, 'NNClassifier')
    create_pc_graph(rf_dict, classifier_path, 'RandomForestClassifier')
    create_pc_graph(svc_dict, classifier_path, 'SVC')
    
    create_roc_graph(knn_dict, classifier_path, 'KNeighborsClassifier')
    create_roc_graph(nn_dict, classifier_path, 'NNClassifier')
    create_roc_graph(rf_dict, classifier_path, 'RandomForestClassifier')
    create_roc_graph(svc_dict, classifier_path, 'SVC')
    
def run_dataset_graphs():
    """Create graph for each dataset"""
    ames_dict = dict()
    bbbp_dict = dict()
    cancer_dict = dict()
    cyp1a2_dict = dict()
    herg_dict = dict()
    hiv_dict =dict()
    liver_dict = dict()
    
    for dataset in DATASETS:
        save_path = os.path.join(save_directory, "datasets", dataset)
        create_dir(save_path)
        for method in AD:
            df_ca = pd.DataFrame()
            df_pc = pd.DataFrame()
            df_roc = pd.DataFrame()
            for classifier in CLASSIFIERS:
                experiment_dir = os.path.join(parent_directory, f"{classifier}_{method}")
                df_ca, df_pc, df_roc = find_mean_dataset(dataset, experiment_dir, df_ca, df_pc, df_roc, name=classifier)
            
            df_ca.to_csv(os.path.join(save_path, f'{method}_ca.csv'), index=False)
            df_pc.to_csv(os.path.join(save_path, f'{method}_pc.csv'), index=False)
            df_roc.to_csv(os.path.join(save_path, f'{method}_roc.csv'), index=False)
            
            if dataset == 'Ames':
                ames_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            elif dataset == 'BBBP':
                bbbp_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            elif dataset == 'Cancer':
                cancer_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            elif dataset == "CYP1A2":
                cyp1a2_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            elif dataset == "hERG":
                herg_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            elif dataset == "HIV":
                hiv_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
            else:
                liver_dict[method] = {"CumulativeAccuracy":df_ca,"PredictivenessCurves": df_pc, "ROC": df_roc}
    
    dataset_path = os.path.join(graph_path, "dataset")
    create_dir(dataset_path)
    
    create_ca_graph(ames_dict, dataset_path, 'Ames')
    create_ca_graph(bbbp_dict, dataset_path, 'BBBP')
    create_ca_graph(cancer_dict, dataset_path, 'Cancer')
    create_ca_graph(cyp1a2_dict, dataset_path, 'CYP1A2')
    create_ca_graph(herg_dict, dataset_path, 'hERG')
    create_ca_graph(hiv_dict, dataset_path, 'HIV')
    create_ca_graph(liver_dict, dataset_path, 'Liver')
    
    create_pc_graph(ames_dict, dataset_path, 'Ames')
    create_pc_graph(bbbp_dict, dataset_path, 'BBBP')
    create_pc_graph(cancer_dict, dataset_path, 'Cancer')
    create_pc_graph(cyp1a2_dict, dataset_path, 'CYP1A2')
    create_pc_graph(herg_dict, dataset_path, 'hERG')
    create_pc_graph(hiv_dict, dataset_path, 'HIV')
    create_pc_graph(liver_dict, dataset_path, 'Liver')
    
    create_roc_graph(ames_dict, dataset_path, 'Ames')
    create_roc_graph(bbbp_dict, dataset_path, 'BBBP')
    create_roc_graph(cancer_dict, dataset_path, 'Cancer')
    create_roc_graph(cyp1a2_dict, dataset_path, 'CYP1A2')
    create_roc_graph(herg_dict, dataset_path, 'hERG')
    create_roc_graph(hiv_dict, dataset_path, 'HIV')
    create_roc_graph(liver_dict, dataset_path, 'Liver')
    
def main():
    run_classifier_graphs()
    run_dataset_graphs()
    
if __name__ == "__main__":
    main()