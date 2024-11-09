import numpy as np
import pandas as pd
import os
import re
import sys
import argparse
import warnings

from syncotrainmp.experiment_setup import current_setup, str_to_bool

# For each round, we need a separate prediction column and a cotrain label.
# The final round only gets a prediction label.

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Semi-Supervised ML for Synthesizability Prediction -- Analyze PU results"
    )
    parser.add_argument(
        "--experiment",
        default="alignn0",
        help="name of the experiment and corresponding config files.",
    )
    parser.add_argument(
        "--ehull015",
        type=str_to_bool,
        default=False,
        help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
    )
    parser.add_argument(
        "--small_data",
        type=str_to_bool,
        default=False,
        help="This option selects a small subset of data for checking the workflow faster.",
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
            # Account for incomplete experiments, when the last prediction is not ready.
            res_dir_list.append(resdir)
            res = pd.read_csv(resdir+'/prediction_results_test_set.csv')
            res_df_list.append(res)
        except:
            pass

    resdf = pd.concat(res_df_list)
    resdf.reset_index(inplace=True, drop=True)
    resdf['material_id'] = resdf['id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])

    return resdf, res_dir_list


def compute_aggregate_df(resdf, propDF, prop, pseudo_label_threshold):
    """Computes an aggregated DataFrame with predictions and labels."""
    agg_df = pd.DataFrame()
    agg_df['avg_prediction'] = resdf.groupby('material_id').prediction.mean()
    agg_df['prediction'] = agg_df['avg_prediction'].map(round)
    agg_df['new_labels'] = agg_df['avg_prediction'].map(lambda x: 1 if x >= pseudo_label_threshold else 0)
    agg_df['target'] = resdf.groupby('material_id').target.first()

    return agg_df.merge(propDF[['material_id', prop]], on='material_id', how='left')


def split_data(agg_df, propDF, prop, id_LOtest):
    """Splits data into experimental, unlabeled, and leave-out test sets."""

    experimental_data = agg_df[agg_df[prop]==1] #needs to change for cotrain?
    unlabeled_data = agg_df[agg_df[prop]==0]   
    
    LO_test = propDF.loc[id_LOtest] 
    LO_test = pd.merge(LO_test, agg_df, on='material_id', how="inner")

    return experimental_data, unlabeled_data, LO_test


def compute_cotrain_labels(propDF, agg_df, TARGET, prop):
    """Computes cotraining labels, filling NaNs with pseudo-labels."""
    cotrain_df = propDF[['material_id', prop]].merge(
    agg_df[['material_id','new_labels']], on='material_id', how='left')

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
        id_LOtest,
        data_prefix,
        ehull015               = False,
        small_data             = False,
        max_iter               = 60,
        pseudo_label_threshold = 0.75,
    ):
    """Generates a performance report for Positive-Unlabeled learning."""

    print(f'experiment is {experiment}, ehull015 is {ehull015} and small data is {small_data}.')

    alignn_dir = "pu_alignn"
    output_dir = os.path.join(alignn_dir, f'PUOutput_{data_prefix}{experiment}')

    if ehull015:
        output_dir = os.path.join(alignn_dir, f'PUehull015_{experiment}')

    propDF = pd.read_pickle(propDFpath)
    resdf, res_dir_list = load_experiment_results(output_dir, max_iter)

    agg_df = compute_aggregate_df(resdf, propDF, prop, pseudo_label_threshold)

    experimental_data = agg_df[agg_df[prop] == 1]
    unlabeled_data    = agg_df[agg_df[prop] == 0]

    experimental_data, unlabeled_data, LO_test = split_data(agg_df, propDF, prop, id_LOtest)

    # Compute statistics
    LO_true_positive_rate = LO_test['prediction'].sum()/len(LO_test)
    true_positive_rate = experimental_data['prediction'].sum()/len(experimental_data)
    predicted_positive_rate = unlabeled_data['prediction'].sum()/len(unlabeled_data)

    cotrain_df = compute_cotrain_labels(propDF, agg_df, TARGET, prop)
       
    report = {'res_dir_list'           : res_dir_list,
              'resdf'                  : resdf,
              'true_positive_rate'     : round(true_positive_rate, 4),
              'LO_true_positive_rate'  : round(LO_true_positive_rate, 4),
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'GT_true_positive_rate'  : '',
              'false_positive_rate'    : '',
              'agg_df'                 : agg_df,
              'cotrain_df'             : cotrain_df }

    if ehull015:
        GT_stable   = propDF[propDF["stability_GT"]==1]
        GT_stable   = pd.merge(GT_stable, agg_df, on='material_id', how="inner")
        GT_unstable = propDF[propDF["stability_GT"]==0]
        GT_unstable = pd.merge(GT_unstable, agg_df, on='material_id', how="inner")
        GT_tpr      = GT_stable['prediction'].sum()/len(GT_stable)
        false_positive_rate = GT_unstable['prediction'].sum()/len(GT_unstable)
        report['GT_true_positive_rate'] = round(GT_tpr, 4)
        report['false_positive_rate'] = round(false_positive_rate, 4)
    
    return report, propDF


def save_results(report, propDF, result_dir, experiment, propDFpath):
    """Saves aggregated and prediction results, updates co-train labels in propDF."""
    filename_agg = os.path.join(result_dir,f'{experiment}.pkl')
    filename_res = os.path.join(result_dir,f'{experiment}_resdf.pkl')

    print(f"Saving aggregated df to `{filename_agg}`")
    report['agg_df'].to_pickle(filename_agg)
    print(f"Saving results df to {filename_res}")
    report['resdf' ].to_pickle(filename_res)

    print(f"Exporting new labels to `{propDFpath}`")
    propDF[experiment] = report['cotrain_df'].new_labels
    propDF.to_pickle(propDFpath)


def save_report(result_dir, experiment, report):
    """Saves final report with performance metrics to a CSV file."""

    filename_report = os.path.join(result_dir, 'results.csv')

    resultcsv = pd.read_csv(filename_report, index_col=0)

    new_rates = {
        'true_positive_rate'     : report['true_positive_rate'],
        'LO_true_positive_rate'  : report['LO_true_positive_rate'],
        'predicted_positive_rate': report['predicted_positive_rate'],
        'GT_true_positive_rate'  : report['GT_true_positive_rate'],
        'false_positive_rate'    : report['false_positive_rate'] }

    print(f"Saving report to {filename_report}")
    resultcsv.loc[experiment.split('PUOutput_')[-1]] = new_rates
    resultcsv.to_csv(filename_report)


def main():
    args = parse_arguments()
    cs, id_LOtest = setup_experiment(args)

    report, propDF = pu_report_alignn(
        experiment  = args.experiment,
        prop        = cs['prop'],
        propDFpath  = cs['propDFpath'],
        TARGET      = cs['TARGET'],
        id_LOtest   = id_LOtest,
        data_prefix = cs['dataPrefix'],
        ehull015    = args.ehull015,
        small_data  = args.small_data)


    print(f"The True positive rate was {report['true_positive_rate']} and the "
        f"predicted positive rate was {report['predicted_positive_rate']}.")

    if args.ehull015:
        print(f"The Groud Truth true-positive-rate was {report['GT_true_positive_rate']} and the "
        f" False positive rate was {report['false_positive_rate']}.")

    save_results(report, propDF, cs['result_dir'], args.experiment, cs['propDFpath'])

    save_report(cs['result_dir'], args.experiment, report)


if __name__ == "__main__":
    main()
