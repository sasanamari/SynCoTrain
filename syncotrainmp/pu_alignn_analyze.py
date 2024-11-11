import numpy as np
import pandas as pd
import os
import re
import sys
import argparse
import warnings

from syncotrainmp.experiment_setup import current_setup

# For each round, we need a separate prediction column and a cotrain label.
# The final round only gets a prediction label.

# TODO: Lots of duplicated code between alignn and schnet

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Semi-Supervised ML for Synthesizability Prediction -- Analyze ALIGNN PU results"
    )
    parser.add_argument(
        "--experiment",
        default="alignn0",
        help="name of the experiment and corresponding config files.",
    )
    parser.add_argument(
        "--ehull015",
        action='store_true',
        default=False,
        help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
    )
    parser.add_argument(
        "--hw",
        action='store_true',
        default=False,
        help="Analysis before the final iteration.",
    )
    parser.add_argument(
        "--small_data",
        action='store_true',
        default=False,
        help="This option selects a small subset of data for checking the workflow faster.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Name of the experiment and corresponding config files."
    )

    return parser.parse_args(sys.argv[1:])


def setup_experiment(args):
    """Sets up the experiment and loads IDs for leave-out test samples."""
    cs = current_setup(small_data=args.small_data, experiment=args.experiment, ehull015 = args.ehull015)

    data_dir = os.path.dirname(cs['propDFpath'])
    split_id_dir = f"{cs['dataPrefix']}{cs['TARGET']}_{cs['prop']}"
    split_id_dir_path = os.path.join(data_dir, split_id_dir)
    LOTestPath = os.path.join(split_id_dir_path, 'leaveout_test_id.txt')

    with open(LOTestPath, "r") as ff:
        id_LOtest = [int(line.strip()) for line in ff]

    return cs, id_LOtest


def load_experiment_results(output_dir, max_iter):
    """Loads prediction results for each iteration until max_iter."""

    res_df_list, res_dir_list = [], []

    for iter, PUiter in enumerate(os.listdir(output_dir)):
        if iter > max_iter:
            break
        resdir = os.path.join(output_dir, PUiter)
        try:
            res_dir_list.append(resdir)
            res = pd.read_csv(resdir+'/prediction_results_test_set.csv')
            res_df_list.append(res)
        except:
            # Account for incomplete experiments, when the last prediction is not ready.
            pass

    resdf = pd.concat(res_df_list)
    resdf.reset_index(inplace=True, drop=True)
    resdf['material_id'] = resdf['id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])

    return resdf, res_dir_list


def compute_aggregate_df(resdf, propDF, prop, pseudo_label_threshold):
    """Computes an aggregated DataFrame with predictions and labels."""
    aggdf = pd.DataFrame()
    aggdf['avg_prediction'] = resdf.groupby('material_id').prediction.mean()
    aggdf['prediction'] = aggdf['avg_prediction'].map(round)
    aggdf['new_labels'] = aggdf['avg_prediction'].map(lambda x: 1 if x >= pseudo_label_threshold else 0)
    aggdf['target'] = resdf.groupby('material_id').target.first()

    return aggdf.merge(propDF[['material_id', prop]], on='material_id', how='left')


def split_data(aggdf, propDF, prop, id_LOtest):
    """Splits data into experimental, unlabeled, and leave-out test sets."""

    experimental_data = aggdf[aggdf[prop] == 1]
    unlabeled_data    = aggdf[aggdf[prop] == 0]
    
    LO_test = propDF.loc[id_LOtest] 
    LO_test = pd.merge(LO_test, aggdf, on='material_id', how="inner")

    return experimental_data, unlabeled_data, LO_test


def compute_cotrain_labels(propDF, aggdf, TARGET, prop):
    """Computes cotraining labels, filling NaNs with pseudo-labels."""
    cotrain_df = propDF[['material_id', prop]].merge(
        aggdf[['material_id','new_labels']], on='material_id', how='left')

    cotrain_index = propDF[propDF[prop]!=propDF[TARGET]].index
    # Used in cotraining, not predicted. does nothing at step 0
    cotrain_df.loc[cotrain_index, 'new_labels'] = 1
    # Filling up the NaN used for training
    cotrain_df.loc[cotrain_df[prop] == 1, 'new_labels'] = 1

    if cotrain_df['new_labels'].isna().mean() > 0.02:
        warnings.warn(f"{round(cotrain_df['new_labels'].isna().mean(), 3) * 100}% of 'new_labels' are NaN.",
                      RuntimeWarning)

    cotrain_df['new_labels'].fillna(cotrain_df[prop], inplace=True)
    cotrain_df.new_labels = cotrain_df.new_labels.astype(np.int16)

    return cotrain_df


def pu_report_alignn(
        experiment: str,
        prop: str,
        propDFpath,
        TARGET,
        data_prefix,
        output_dir,
        id_LOtest,
        ehull015               = False,
        small_data             = False,
        max_iter               = 60,
        pseudo_label_threshold = 0.75,
    ):
    """Generates a performance report for Positive-Unlabeled learning."""

    print(f'experiment is {experiment}, ehull015 is {ehull015} and small data is {small_data}.')

    alignn_dir = "pu_alignn"
    output_dir = os.path.join(output_dir, alignn_dir, f'PUOutput_{data_prefix}{experiment}')

    if ehull015:
        output_dir = os.path.join(alignn_dir, f'PUehull015_{experiment}')

    propDF = pd.read_pickle(propDFpath)
    resdf, res_dir_list = load_experiment_results(output_dir, max_iter)

    aggdf = compute_aggregate_df(resdf, propDF, prop, pseudo_label_threshold)

    experimental_data, unlabeled_data, LO_test = split_data(aggdf, propDF, prop, id_LOtest)

    # Compute statistics
    LO_true_positive_rate = LO_test['prediction'].mean()
    true_positive_rate = experimental_data['prediction'].mean()
    predicted_positive_rate = unlabeled_data['prediction'].mean()

    cotraindf = compute_cotrain_labels(propDF, aggdf, TARGET, prop)
       
    report = {'res_dir_list'           : res_dir_list,
              'resdf'                  : resdf,
              'agg_df'                 : aggdf,
              'cotrain_df'             : cotraindf,
              'true_positive_rate'     : round(true_positive_rate, 4),
              'LO_true_positive_rate'  : round(LO_true_positive_rate, 4),
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'GT_true_positive_rate'  : '',
              'false_positive_rate'    : '' }

    if ehull015:

        GT_stable           = propDF[propDF["stability_GT"]==1]
        GT_stable           = pd.merge(GT_stable, aggdf, on='material_id', how="inner")
        GT_unstable         = propDF[propDF["stability_GT"]==0]
        GT_unstable         = pd.merge(GT_unstable, aggdf, on='material_id', how="inner")
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
    resultcsv.loc[experiment] = new_rates
    resultcsv.to_csv(filename_results)


def main():
    args = parse_arguments()
    cs, id_LOtest = setup_experiment(args)

    report, propDF = pu_report_alignn(
        args.experiment,
        cs['prop'],
        cs['propDFpath'],
        cs['TARGET'],
        cs['dataPrefix'],
        args.output_dir,
        id_LOtest,
        ehull015    = args.ehull015,
        small_data  = args.small_data)

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
