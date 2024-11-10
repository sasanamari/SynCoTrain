# %%
import os
import json
import pandas as pd
import numpy as np
import sys
import argparse

from syncotrainmp.experiment_setup import current_setup, str_to_bool

# TODO: Lots of duplicated code between alignn and schnet

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Semi-Supervised ML for Synthesizability Prediction -- Analyze SchNet PU results"
    )
    parser.add_argument(
        "--experiment",
        default="schnet0",
        help="name of the experiment and corresponding config files.",
    )
    parser.add_argument(
        "--ehull015",
        type=str_to_bool,
        default=False,
        help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
    )
    parser.add_argument(
        "--hw",
        type=str_to_bool,
        default=False,
        help="Analysis before the final iteration.",
    )
    parser.add_argument(
        "--startIt", 
        type=int, 
        default=0, 
        help="Starting iteration No.")
    parser.add_argument(
        "--small_data",
        type=str_to_bool,
        default=False,
        help="This option selects a small subset of data for checking the workflow faster.",
    )
    return parser.parse_args(sys.argv[1:])


def setup_experiment(args):
    """Sets up the experiment and loads IDs for leave-out test samples."""
    cs = current_setup(small_data=args.small_data, experiment=args.experiment, ehull015=args.ehull015)

    data_dir = os.path.dirname(cs['propDFpath'])
    split_id_dir = f"{cs['dataPrefix']}{cs['TARGET']}_{cs['prop']}"
    split_id_dir_path = os.path.join(data_dir, split_id_dir)
    LOTestPath = os.path.join(split_id_dir_path, 'leaveout_test_id.txt')

    with open(LOTestPath, "r") as ff:
        id_LOtest = [int(line.strip()) for line in ff] 

    return cs, id_LOtest


def score_function(x):
    """Calculate score as a percentage of positive predictions and count the number of trials."""
    trial_num = sum(x.notna())

    if trial_num == 0:
        return np.nan, trial_num
    else:
        return x.sum()/trial_num, trial_num


def load_config(small_data=False):
    schnet_config_dir = "pu_schnet/schnet_configs"
    config_path = os.path.join(schnet_config_dir, 'pu_config_schnetpack.json')
        
    with open(config_path, "r") as read_file:
        print("Read Experiment configuration")
        config = json.load(read_file)

    if small_data:
        config["epoch_num"] = int(config["epoch_num"]*0.5)

    return config


def load_experiment_results(config, data_prefix, experiment, max_iter, ehull015, half_way_analysis, startIt):

    resdf_filename = f'{data_prefix}{experiment}_{str(startIt)}_{str(config["num_iter"])}ep{str(config["epoch_num"])}'

    schnet_dir = config["schnetDirectory"]

    if ehull015:
        output_dir = os.path.join(schnet_dir, f'PUehull015_{data_prefix}{experiment}')
    else:
        output_dir = os.path.join(schnet_dir, f'PUOutput_{data_prefix}{experiment}')

    if half_way_analysis:
        resdf = pd.read_pickle(os.path.join(output_dir, 'res_df', f'{resdf_filename}tmp'))
    else:
        resdf = pd.read_pickle(os.path.join(output_dir, 'res_df', resdf_filename))
   
    pred_columns = []
    excess_iters = []

    # Always start at 0 because we want to average prediction over all the iterations.
    for it in range(0, config["num_iter"]):
        pred_col_name = 'pred_'+str(it)
        if half_way_analysis:
            if pred_col_name not in resdf.columns:
                continue
        if it > max_iter:
            excess_iters.append(pred_col_name)
        else:            
            pred_columns.append(pred_col_name)

    return resdf, pred_columns, excess_iters


def split_data(resdf, propDF, prop, id_LOtest):
    """Splits data into experimental, unlabeled, and leave-out test sets."""

    experimental_data = resdf[resdf[prop] == 1]
    unlabeled_data    = resdf[resdf[prop] == 0]

    LO_test = propDF.loc[id_LOtest] 
    LO_test = pd.merge(LO_test, resdf, on='material_id', how="inner")

    return experimental_data, unlabeled_data, LO_test


def pu_report_schnet(
        experiment: str,
        prop: str,
        propDFpath,
        TARGET,
        data_prefix,
        id_LOtest,
        ehull015               = False,
        small_data             = False,
        half_way_analysis      = False,
        startIt                = 0,
        max_iter               = 60,
        pseudo_label_threshold = 0.75,
    ):

    print(f'experiment is {experiment}, ehull015 is {ehull015} and small data is {small_data}.')

    config = load_config(small_data=small_data)

    propDF = pd.read_pickle(propDFpath)
    resdf, pred_columns, excess_iters = load_experiment_results(config, data_prefix, experiment, max_iter, ehull015, half_way_analysis, startIt)

    Preds = resdf[pred_columns]
    resdf['predScore'] = Preds.apply(score_function, axis=1)
    resdf[['predScore', 'trial_num']] = pd.DataFrame(resdf.predScore.tolist())
    resdf["prediction"] = resdf.predScore.map(lambda x: x if np.isnan(x) else round(x))
    resdf["new_labels"] = resdf.predScore.map(lambda x: x if np.isnan(x) else 1 if x >= pseudo_label_threshold else 0)

    resdf = resdf[resdf.predScore.notna()][[
        'material_id', prop, TARGET,'prediction', 'predScore', 'trial_num']]  #selecting data with prediction values
    resdf = resdf.loc[:, ~resdf.columns.duplicated()] # drops duplicated props at round zero.

    resdf = resdf.drop(columns=Preds)
    resdf = resdf.drop(columns=excess_iters) #might require a df like Preds    
    
    experimental_data, unlabeled_data, LO_test = split_data(resdf, propDF, prop, id_LOtest)

    # Compute statistics
    LO_true_positive_rate = LO_test.prediction.mean()
    true_positive_rate = experimental_data.prediction.mean()
    predicted_positive_rate = unlabeled_data.prediction.mean()

    print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, config["num_iter"], config["epoch_num"]))
    print('Our LO true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(LO_true_positive_rate*100, config["num_iter"], config["epoch_num"]))
    print('and {:.1f}% of currenly unlabeled data have been predicted to belong to the positive class.'.format(predicted_positive_rate*100))    
    
    merged_df = propDF[['material_id', prop]].merge(
        unlabeled_data[['material_id', 'prediction']], on='material_id', how='left')
    merged_df = merged_df.rename(columns={'prediction': 'new_labels'}) #for clarity

    # The data used in training will have nan values. We simplify below and check in practice.
    cotrain_index = propDF[(propDF[prop] != propDF[TARGET])].index
    merged_df.loc[cotrain_index, 'new_labels'] = 1
  
    merged_df.loc[merged_df[prop] == 1, 'new_labels'] = 1

    merged_df.new_labels = merged_df.new_labels.astype(pd.Int16Dtype())
    
    propDF[experiment] = merged_df.new_labels
    
    report = {'resdf'                  : Preds,
              'agg_df'                 : resdf, 
              'true_positive_rate'     : round(true_positive_rate, 4), 
              'LO_true_positive_rate'  : round(LO_true_positive_rate, 4),
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'GT_true_positive_rate'  : '',
              'false_positive_rate'    : '' }

    if ehull015:

        GT_stable           = propDF[propDF["stability_GT"]==1] 
        GT_stable           = pd.merge(GT_stable, resdf, on='material_id', how="inner")
        GT_unstable         = propDF[propDF["stability_GT"]==0]
        GT_unstable         = pd.merge(GT_unstable, resdf, on='material_id', how="inner") 
        GT_tpr              = GT_stable['prediction'].mean()
        false_positive_rate = GT_unstable['prediction'].mean()

        report['GT_true_positive_rate'] = round(GT_tpr, 4)
        report['false_positive_rate']   = round(false_positive_rate, 4)

        print(f"The Groud Truth true-positive-rate was {report['GT_true_positive_rate']*100}% and the")
        print(f"False positive rate was {100*report['false_positive_rate']}%.")
        
    return report, propDF


def save_report(report, result_dir, experiment):
    """Saves aggregated and prediction results, updates co-train labels in propDF."""
    filename_agg = os.path.join(result_dir,f'{experiment}.pkl')
    filename_res = os.path.join(result_dir,f'{experiment}_resdf.pkl')

    print(f"Saving aggregated df to `{filename_agg}`")
    report['agg_df'].to_pickle(filename_agg)
    print(f"Saving results df to {filename_res}")
    report['resdf' ].to_pickle(filename_res)


def save_predictions(report, propDF, experiment, propDFpath):
    """Export predictions to propDF"""
    print(f"Exporting new labels to `{propDFpath}`")
    propDF[experiment] = report['cotrain_df'].new_labels
    propDF.to_pickle(propDFpath)


def update_results_csv(report, result_dir, experiment):
    """Updates the results.csv file with the latest experiment report metrics."""

    filename_results = os.path.join(result_dir, 'results.csv')

    resultcsv = pd.read_csv(filename_results, index_col=0)

    new_rates = {
        'true_positive_rate'     : report['true_positive_rate'],
        'LO_true_positive_rate'  : report['LO_true_positive_rate'],
        'predicted_positive_rate': report['predicted_positive_rate'],
        'GT_true_positive_rate'  : report['GT_true_positive_rate'],
        'false_positive_rate'    : report['false_positive_rate'] }

    print(f"Saving report to {filename_results}")
    resultcsv.loc[experiment.split('PUOutput_')[-1]] = new_rates
    resultcsv.to_csv(filename_results)


def main():
    args = parse_arguments()
    cs, id_LOtest = setup_experiment(args)

    report, propDF = pu_report_schnet(
        args.experiment,
        cs['prop'],
        cs['propDFpath'],
        cs['TARGET'],
        cs['dataPrefix'],
        id_LOtest,
        startIt           = args.startIt,
        ehull015          = args.ehull015, 
        small_data        = args.small_data,
        half_way_analysis = args.hw
    )

    print(f"The True positive rate was {report['true_positive_rate']} and the "
        f"predicted positive rate was {report['predicted_positive_rate']}.")

    if args.ehull015:
        print(f"The Groud Truth true-positive-rate was {report['GT_true_positive_rate']} and the "
        f" False positive rate was {report['false_positive_rate']}.")

    if args.hw:
        return

    save_report(report, cs['result_dir'], args.experiment)

    save_predictions(report, propDF, args.experiment, cs['propDFpath'])

    update_results_csv(report, cs['result_dir'], args.experiment)


if __name__ == "__main__":
    main()
