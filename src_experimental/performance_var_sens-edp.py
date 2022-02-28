"""
Variance-based sensitivity analysis of the FEMA P-58 methodology
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm


idx = pd.IndexSlice

logging.basicConfig(
    filename=None,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)


logging.info('Start')
logging.info('Defining functions and global variables')

response_path = 'analysis/hazard_level_3/response_summary/response.csv'
c_modeling_uncertainty = 0.00
num_realizations = 100000
replacement_threshold = 0.50
perf_model_input_path = 'src/new_perf/input_cmp_quant.csv'
cmp_fragility_input_path = 'src/new_perf/input_fragility.csv'
cmp_repair_cost_input_path = 'src/new_perf/input_repair_cost.csv'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       FUNCTIONS                                #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def lognormal_transform(unif_sample, delta, beta):
    return np.exp(norm.ppf(unif_sample, loc=np.log(delta), scale=beta))


def normal_transform(unif_sample, mean, stdev):
    return norm.ppf(unif_sample, loc=mean, scale=stdev)


# EDPs

def gen_edp_samples(resp_path, num_rlz, c_mdl):
    data = pd.read_csv(
        resp_path, header=0, index_col=0,
        low_memory=False)
    data.drop('units', inplace=True)
    data = data.astype(float)
    index_labels = [label.split('-') for label in data.columns]
    index_labels = np.array(index_labels)
    data.columns = pd.MultiIndex.from_arrays(index_labels.T)
    raw_data = np.log(data.to_numpy())
    mean_vec = np.mean(raw_data, axis=0)
    covariance_mat = np.cov(raw_data.T)
    correl_mat = np.corrcoef(raw_data.T)
    sigma2_vec = np.diag(covariance_mat)
    sigma2_inflated = sigma2_vec + c_mdl**2
    diagonal_mat = np.diag(np.sqrt(sigma2_inflated))
    covariance_mat = diagonal_mat @ correl_mat @ diagonal_mat
    z_mat = np.random.multivariate_normal(
        mean_vec, covariance_mat, size=num_rlz)
    samples_raw = np.exp(z_mat)
    samples = pd.DataFrame(samples_raw, index=None)
    samples.columns = data.columns
    samples.index.name = 'realization'
    samples.sort_index(axis=1, inplace=True)
    return samples


# component quantities

def read_perf_model(perf_model_input_path):

    perf_model_input = pd.read_csv(perf_model_input_path, index_col=None)
    # expanded dataframe with one group per line
    perf_model_input = perf_model_input.astype(
        {'location': str, 'direction': str, 'mean': str})
    perf_model = pd.DataFrame(columns=perf_model_input.columns)
    series_list = []
    for i, entry in perf_model_input.iterrows():
        locations = entry['location'].split(',')
        directions = entry['direction'].split(',')
        means = entry['mean'].split(',')
        for loc in locations:
            for drc in directions:
                for j, mean in enumerate(means):
                    row = pd.Series(
                        [entry['component'],
                         entry['edp type'],
                         int(loc), int(drc),
                         int(j), float(mean),
                         float(entry['cov']),
                         entry['distribution'],
                         entry['unit'],
                         entry['description']],
                        index=['component',
                               'edp type',
                               'location',
                               'direction',
                               'group',
                               'mean',
                               'cov',
                               'distribution',
                               'unit',
                               'description'
                               ])
                    series_list.append(row)
    perf_model = pd.concat(series_list, axis=1).transpose()
    col_idx = []
    for i, entry in perf_model.iterrows():
        col_idx.append([entry['component'], entry['location'],
                        entry['direction'], entry['group']])
    col_idx = pd.MultiIndex.from_arrays(np.array(col_idx).T)
    perf_model.index = col_idx
    perf_model.drop(['component', 'location', 'direction', 'group'],
                    axis=1, inplace=True)
    perf_model.index.name = 'component group'
    perf_model.sort_index(axis=0, inplace=True)
    del(perf_model_input)
    return perf_model


def gen_cmp_quant_RV(perf_model):

    # initialize empty dataframe
    cmp_quant_RV = pd.DataFrame(
        np.full((num_realizations, perf_model.shape[0]), 0.00),
        columns=perf_model.index, index=None)
    cmp_quant_RV.columns.names = [
        'component', 'location', 'direction', 'group']
    cmp_quant_RV.index.name = 'realizations'
    cmp_quant_RV.sort_index(axis=1, inplace=True)

    # filter components with fixed quantities
    no_distr_idx = perf_model[perf_model['distribution'].isnull()].index
    for comp_group in no_distr_idx:
        cmp_quant_RV.loc[:, comp_group] = perf_model.loc[comp_group, 'mean']

    # filter components with lognormal quantities
    lognormal_idx = perf_model[perf_model['distribution'] == 'lognormal'].index
    assert(len(lognormal_idx) + len(no_distr_idx) == perf_model.shape[0]), \
        "Only lognormal is supported here."
    for comp_group in lognormal_idx:
        mu = perf_model.loc[comp_group, 'mean']
        cov = perf_model.loc[comp_group, 'cov']
        sgm = mu * cov
        mu_n = np.log(mu) - 0.50 * np.log(1. + (sgm/mu)**2)
        sgm_n = np.sqrt(np.log(1. + (sgm/mu)**2))
        vec = np.exp(np.random.normal(mu_n, sgm_n, num_realizations))
        cmp_quant_RV.loc[:, comp_group] = vec
    return cmp_quant_RV


# component damage

def read_fragility_input(cmp_fragility_input_path):
    cmp_fragility_input = pd.read_csv(
        cmp_fragility_input_path, index_col=0)
    cmp_fragility_input.sort_index(inplace=True)
    return cmp_fragility_input


def gen_cmp_fragility_RV(cmp_fragility_input, perf_model):
    # instantiate a dataframe with the right indices
    all_groups = perf_model.index.values
    comp_ids_uniq = set()
    for group in all_groups:
        comp_ids_uniq.add(group[0])
    comp_ids_uniq = list(comp_ids_uniq)
    comp_ids_uniq.sort()
    col_idx = []
    for comp_id in comp_ids_uniq:
        list_of_these_groups = []
        for group in all_groups:
            if group[0] == comp_id:
                list_of_these_groups.append(group)
        for i in range(3):
            delta = cmp_fragility_input.loc[comp_id, f'DS{i+1}_delta']
            if pd.isna(delta):
                continue
            col_idx.extend([(*item, f'DS{i+1}')
                            for item in list_of_these_groups])
    cmp_fragility_RV = pd.DataFrame(index=range(num_realizations),
                                    columns=pd.MultiIndex.from_tuples(col_idx))
    cmp_fragility_RV.columns.names = ['component', 'location', 'direction',
                                      'group', 'damage state']
    cmp_fragility_RV.sort_index(axis=1, inplace=True)
    cmp_fragility_RV.sort_index(axis=0, inplace=True)
    # sample damage state threshold values
    for group in all_groups:
        comp_id = group[0]
        unif_sample = np.random.uniform(0.00, 1.00, num_realizations)
        for i in range(3):
            delta = cmp_fragility_input.loc[comp_id, f'DS{i+1}_delta']
            if pd.isna(delta):
                continue
            beta = cmp_fragility_input.loc[comp_id, f'DS{i+1}_beta']
            sample = lognormal_transform(unif_sample, delta, beta)
            sample *= (delta / np.median(sample))
            cmp_fragility_RV.loc[:, (*group, f'DS{i+1}')] = sample

    # debug - fix to the mean
    cols = cmp_fragility_RV.columns
    cols = cols[:-2]  # remove 'collapse, 'irreparable'
    cmp_fragility_RV.loc[:, cols] = cmp_fragility_RV.loc[:, cols].mean(axis=0).to_numpy()
    return cmp_fragility_RV


def calc_cmp_damage(perf_model, cmp_fragility_input,
                    edp_RV, cmp_fragility_RV):
    all_groups = perf_model.index.values
    cmp_damage = pd.DataFrame(
        np.full((num_realizations, len(all_groups)), 0, dtype=np.int32),
        index=range(num_realizations),
        columns=pd.MultiIndex.from_tuples(all_groups))
    cmp_damage.columns.names = ['component', 'location', 'direction', 'group']
    cmp_damage.sort_index(axis=1, inplace=True)
    cmp_damage.sort_index(axis=0, inplace=True)
    for group in all_groups:
        fragility_info = cmp_fragility_input.loc[group[0], :]
        if fragility_info.directional == 1:
            edp_vec = edp_RV.loc[
                :, (fragility_info['edp type'], group[1], group[2])]
        else:
            edp_vec = edp_RV.loc[
                :, (fragility_info['edp type'], group[1])].max(axis=1) * 1.2
        for i in range(3):
            delta = cmp_fragility_input.loc[group[0], f'DS{i+1}_delta']
            if pd.isna(delta):
                continue
            ds_mask = edp_vec > cmp_fragility_RV.loc[:, (*group, f'DS{i+1}')]
            cmp_damage.loc[ds_mask, group] = i+1
    # "damage process"
    # reaching a specified damage state of particular components
    # trigger damage states of other components
    # when D2021.011a reaches DS2 --> C3021.001k DS1 is enabled
    cols = cmp_damage.loc[:, ('D2021.011a')].columns
    for col in cols:
        loc, drct, group = col
        sub = cmp_damage.loc[:, ('D2021.011a', *col)]
        indx = sub[sub == 2].index
        cmp_damage.loc[indx, ('C3021.001k', str(int(loc)-1), drct, group)] = 1

    # when D4011.021a reaches DS2 --> C3021.001k DS1 is enabled
    cols = cmp_damage.loc[:, ('D4011.021a')].columns
    for col in cols:
        loc, drct, group = col
        sub = cmp_damage.loc[:, ('D2021.011a', *col)]
        indx = sub[sub == 2].index
        cmp_damage.loc[indx, ('C3021.001k', str(int(loc)-1), drct, group)] = 1
    return cmp_damage


# component repair cost

def read_cmp_repair_cost_input(cmp_repair_cost_input_path):
    cmp_repair_cost_input = pd.read_csv(
        cmp_repair_cost_input_path, index_col=0)
    cmp_repair_cost_input.drop(['description', 'note'], inplace=True, axis=1)

    def pipe_split(x, side):
        if pd.isna(x):
            return pd.NA
        else:
            if '|' in x:
                res = x.split('|')[side]
                if ',' in res:
                    res = res.split(',')
                    res = [float(x) for x in res]
                else:
                    res = [float(res), float(res)]
                return res
            else:
                if side == 0:
                    return [float(x), float(x)]
                else:
                    return [2., 1.]

    damage_consequences = ['DS1_1', 'DS1_2', 'DS2_1', 'DS3_1']
    for dc in damage_consequences:
        cmp_repair_cost_input[f'{dc}_theta0_c'] = \
            cmp_repair_cost_input[f'{dc}_theta0'].apply(pipe_split, side=0)
        cmp_repair_cost_input[f'{dc}_theta0_q'] = \
            cmp_repair_cost_input[f'{dc}_theta0'].apply(pipe_split, side=1)
        cmp_repair_cost_input.drop(f'{dc}_theta0', inplace=True, axis=1)
    return cmp_repair_cost_input


def num_DS(cmp_repair_cost_input, comp_id):
    nds = 0
    for i in range(3):
        if not np.isnan(cmp_repair_cost_input.loc[comp_id][f'DS{i+1}_n']):
            nds += 1
    return nds


def gen_cmp_damage_consequence_RV(perf_model, cmp_repair_cost_input):
    # pick damage state consequence
    # generate columns
    cols = []
    all_groups = perf_model.index.values
    for group in all_groups:
        comp_id = group[0]
        if comp_id in ['collapse', 'irreparable']:
            continue
        nds = num_DS(cmp_repair_cost_input, comp_id)
        for i_s in range(nds):
            cols.append((*group, f'DS{i_s+1}'))
    cols.append(('replacement', '0', '1', '0', 'DS1'))
    col_idx = pd.MultiIndex.from_tuples(cols)
    col_idx.names = ['component', 'location', 'direction',
                     'group', 'damage state']
    cmp_damage_consequence_RV = pd.DataFrame(
        columns=col_idx, index=range(num_realizations))
    cmp_damage_consequence_RV.index.name = 'realizations'
    # ds_len = [2, 1, 1]
    # shortcut: we know that only the first damage state has two
    # possible consequences
    cmp_damage_consequence_RV.loc[:, idx[:, :, :, :, ['DS2', 'DS3']]] = 1
    comp_ids_uniq = set()
    for group in all_groups:
        comp_ids_uniq.add(group[0])
    comp_ids_uniq = list(comp_ids_uniq)
    comp_ids_uniq.sort()
    comp_ids_uniq.remove('collapse')
    comp_ids_uniq.remove('irreparable')
    comp_ids_uniq.append('replacement')
    for comp_id in comp_ids_uniq:
        w1 = cmp_repair_cost_input.loc[comp_id, :]['DS1_1_w']
        w2 = cmp_repair_cost_input.loc[comp_id, :]['DS1_2_w']
        if not np.isnan(w2):
            assert(w1 + w2 - 1.00 < 1e-5)
            n_cols = cmp_damage_consequence_RV.loc[
                :, idx[comp_id, :, :, :, 'DS1']].shape[1]
            mat = np.random.binomial(1, w2, (num_realizations, n_cols)) + 1
            cmp_damage_consequence_RV.loc[
                :, idx[comp_id, :, :, :, 'DS1']] = mat
        else:
            cmp_damage_consequence_RV.loc[:, idx[comp_id, :, :, :, 'DS1']] = 1
    return cmp_damage_consequence_RV


def calc_cmp_dmg_quant(perf_model, cmp_repair_cost_input, cmp_damage,
                       cmp_damage_consequence_RV, cmp_quant_RV):
    # generate columns
    cols = []
    # ds_len = [2, 1, 1]
    all_groups = perf_model.index.values
    for group in all_groups:
        comp_id = group[0]
        if comp_id in ['collapse', 'irreparable']:
            continue
        nds = num_DS(cmp_repair_cost_input, comp_id)
        for i_s in range(nds):
            n_cons = int(cmp_repair_cost_input.loc[comp_id][f'DS{i_s+1}_n'])
            for consequence in range(n_cons):
                cols.append((*group, f'DS{i_s+1}', str(consequence + 1)))
    cols.append(('replacement', '0', '1', '0', 'DS1', '1'))
    col_idx = pd.MultiIndex.from_tuples(cols)
    col_idx.names = ['component', 'location', 'direction',
                     'group', 'damage state', 'consequence']
    # gather damaged quantities
    cmp_dmg_quant = pd.DataFrame(
        np.zeros((num_realizations, len(col_idx))),
        columns=col_idx, index=range(num_realizations))
    cmp_dmg_quant.index.name = 'realizations'
    cmp_dmg_quant.sort_index(axis=1, inplace=True)
    for i_col, col in enumerate(cmp_damage.columns):
        if col[0] in ['collapse', 'irreparable']:
            continue
        vec = cmp_damage.iloc[:, i_col]
        nds = num_DS(cmp_repair_cost_input, col[0])
        n_cons = int(cmp_repair_cost_input.loc[
            col[0]]['DS1_n'])
        idx_ds1 = vec[vec == 1].index
        idx_ds1_1 = cmp_damage_consequence_RV.loc[idx_ds1, (*col, 'DS1')][
            cmp_damage_consequence_RV.loc[idx_ds1, (*col, 'DS1')] == 1].index
        cmp_dmg_quant.loc[idx_ds1_1, (*col, 'DS1', '1')] = \
            cmp_quant_RV.loc[idx_ds1_1, col]
        if n_cons > 1:
            idx_ds1_2 = cmp_damage_consequence_RV.loc[idx_ds1, (*col, 'DS1')][
                cmp_damage_consequence_RV.loc[
                    idx_ds1, (*col, 'DS1')] == 2].index
            cmp_dmg_quant.loc[idx_ds1_2, (*col, 'DS1', '2')] = \
                cmp_quant_RV.loc[idx_ds1_2, col]
        if nds > 1:
            idx_ds2 = vec[vec == 2].index
            cmp_dmg_quant.loc[idx_ds2, (*col, 'DS2', '1')] = \
                cmp_quant_RV.loc[idx_ds2, col]
        if nds > 2:
            idx_ds3 = vec[vec == 3].index
            cmp_dmg_quant.loc[idx_ds3, (*col, 'DS3', '1')] = \
                cmp_quant_RV.loc[idx_ds3, col]
    for i_col, col in enumerate(cmp_damage.columns):
        if col[0] in ['collapse', 'irreparable']:
            vec = cmp_damage.iloc[:, i_col]
            indx = vec[vec == 1].index
            cmp_dmg_quant.loc[indx, :] = 0.00
            cmp_dmg_quant.loc[
                indx, ('replacement', '0', '1', '0', 'DS1', '1')] = \
                cmp_quant_RV.loc[indx, col]
    # assuming economy of scale, we add the damaged quantities
    # across location, direction, and group.
    cmp_dmg_quant_eco = cmp_dmg_quant.groupby(
        level=['component'], axis=1).sum()
    return cmp_dmg_quant, cmp_dmg_quant_eco


def gen_cmp_cost_RV(cmp_dmg_quant):
    cmp_cost_RV = pd.DataFrame(
        np.random.uniform(
            0.00, 1.00, (num_realizations, len(cmp_dmg_quant.columns))),
        columns=cmp_dmg_quant.columns, index=range(num_realizations))
    cmp_cost_RV.index.name = 'realizations'
    cmp_cost_RV.sort_index(axis=1, inplace=True)
    return cmp_cost_RV


def calc_cmp_cost(cmp_dmg_quant, cmp_cost_RV,
                  cmp_repair_cost_input, cmp_dmg_quant_eco):
    cmp_cost = pd.DataFrame(
        np.zeros((num_realizations, len(cmp_dmg_quant.columns))),
        columns=cmp_dmg_quant.columns,
        index=cmp_cost_RV.index)
    cmp_cost.index.name = 'realizations'
    cmp_cost.sort_index(axis=1, inplace=True)
    # (this could be made more efficient if it was looping over
    #  components instead of looping over groups)
    # but the result is the same
    for i_col, col in enumerate(cmp_cost_RV.columns):
        comp_id, location, direction, group, damage_state, consequence = col
        b_q = cmp_repair_cost_input.loc[comp_id, :]['base_quantity']
        median_dmg_quant = cmp_dmg_quant_eco.loc[:, comp_id] / b_q
        es_x = cmp_repair_cost_input.loc[
            comp_id, :][f'{damage_state}_{consequence}_theta0_q']
        es_y = cmp_repair_cost_input.loc[
            comp_id, :][f'{damage_state}_{consequence}_theta0_c']
        cost = np.zeros(num_realizations)
        sel = median_dmg_quant < es_x[0]
        cost[sel] = es_y[0]
        sel = median_dmg_quant > es_x[1]
        cost[sel] = es_y[1]
        sel = np.logical_and(median_dmg_quant <= es_x[1],
                             median_dmg_quant >= es_x[0])
        cost[sel] = (es_y[1]-es_y[0])/(es_x[1]-es_x[0])*(
            median_dmg_quant[sel]-es_x[0])+es_y[0]
        # add variability
        distribution = cmp_repair_cost_input.loc[comp_id][
            f'{damage_state}_{consequence}_distribution']
        if distribution == 'normal':
            cov = cmp_repair_cost_input.loc[comp_id, :][
                f'{damage_state}_{consequence}_theta1']
            sample = normal_transform(cmp_cost_RV.loc[:, col], 1.00, cov)
            sample /= np.mean(sample)
            cmp_cost.loc[:, col] = cmp_dmg_quant.loc[
                :, col] / b_q * cost * sample
        elif distribution == 'lognormal':
            beta = cmp_repair_cost_input.loc[comp_id, :][
                f'{damage_state}_{consequence}_theta1']
            sample = lognormal_transform(cmp_cost_RV.loc[:, col], 1.00, beta)
            sample /= np.median(sample)
            cmp_cost.loc[:, col] = cmp_dmg_quant.loc[
                :, col] / b_q * cost * sample
            # cmp_cost.loc[:, col] = cost
        elif distribution == 'zero':
            cmp_cost.loc[:, col] = 0.00
        else:
            raise ValueError(
                f'Unknown distribution encountered: {distribution}')
    return cmp_cost


# summarize cost

def calc_total_cost(cmp_cost, cmp_repair_cost_input, cmp_cost_RV):
    total_cost = cmp_cost.sum(axis=1)
    # consider replacement cost threshold value
    col = ('replacement', '0', '1', '0', 'DS1', '1')
    rpl = cmp_cost.loc[:, col]
    non_replacement_idx = rpl[rpl == 0.00].index
    n_samples = len(non_replacement_idx)
    if n_samples != 0:
        delta = cmp_repair_cost_input.loc['replacement', 'DS1_1_theta0_c'][0]
        beta = cmp_repair_cost_input.loc['replacement', 'DS1_1_theta1']
        mean = delta * np.exp(beta**2/2.)
        change_idx = total_cost.loc[
            non_replacement_idx][
                total_cost.loc[
                    non_replacement_idx] > replacement_threshold * mean].index
        sample = lognormal_transform(
            cmp_cost_RV.loc[change_idx, col], delta, beta)
        sample *= delta/np.median(sample)
        total_cost[change_idx] = sample
    return total_cost


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                    Reading Input Data                          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

perf_model_df = read_perf_model(perf_model_input_path)
cmp_fragility_input_df = read_fragility_input(cmp_fragility_input_path)
cmp_repair_cost_input_df = \
    read_cmp_repair_cost_input(cmp_repair_cost_input_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       Analysis A                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

logging.info('Starting analysis A')
analysis_A = {}
logging.info('\tGenerating simulated demands')
analysis_A['edp_RV'] = gen_edp_samples(response_path, num_realizations,
                                       c_modeling_uncertainty)
logging.info('\tSampling component quantities')
analysis_A['cmp_quant_RV_df'] = gen_cmp_quant_RV(perf_model_df)
logging.info('\tSampling component damage state thresholds')
analysis_A['cmp_fragility_RV_df'] = gen_cmp_fragility_RV(
    cmp_fragility_input_df, perf_model_df)
logging.info('\tDetermining component damage')
analysis_A['cmp_damage_df'] = calc_cmp_damage(
    perf_model_df, cmp_fragility_input_df,
    analysis_A['edp_RV'],
    analysis_A['cmp_fragility_RV_df'])
logging.info('\tDetermining damage consequences')
analysis_A['cmp_damage_consequence_RV_df'] = gen_cmp_damage_consequence_RV(
    perf_model_df, cmp_repair_cost_input_df)
analysis_A['cmp_dmg_quant_df'], analysis_A['cmp_dmg_quant_eco_df'] = \
    calc_cmp_dmg_quant(
        perf_model_df, cmp_repair_cost_input_df,
        analysis_A['cmp_damage_df'],
        analysis_A['cmp_damage_consequence_RV_df'],
        analysis_A['cmp_quant_RV_df'])
logging.info('\tDetermining component repair cost')
analysis_A['cmp_cost_RV_df'] = gen_cmp_cost_RV(
    analysis_A['cmp_dmg_quant_df'])
analysis_A['cmp_cost_df'] = calc_cmp_cost(
    analysis_A['cmp_dmg_quant_df'],
    analysis_A['cmp_cost_RV_df'],
    cmp_repair_cost_input_df,
    analysis_A['cmp_dmg_quant_eco_df'])
logging.info('\tSummarizing cost')
analysis_A['total_cost_df'] = calc_total_cost(
    analysis_A['cmp_cost_df'],
    cmp_repair_cost_input_df,
    analysis_A['cmp_cost_RV_df'])
logging.info('\tAnalysis A finished')


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# #                       Analysis B                               #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# logging.info('Starting analysis B')
# analysis_B = {}
# logging.info('\tGenerating simulated demands')
# analysis_B['edp_RV'] = gen_edp_samples(response_path, num_realizations,
#                                        c_modeling_uncertainty)
# logging.info('\tSampling component quantities')
# analysis_B['cmp_quant_RV_df'] = gen_cmp_quant_RV(perf_model_df)
# logging.info('\tSampling component damage state thresholds')
# analysis_B['cmp_fragility_RV_df'] = gen_cmp_fragility_RV(
#     cmp_fragility_input_df, perf_model_df)
# logging.info('\tDetermining component damage')
# analysis_B['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_B['edp_RV'],
#     analysis_B['cmp_fragility_RV_df'])
# logging.info('\tDetermining damage consequences')
# analysis_B['cmp_damage_consequence_RV_df'] = gen_cmp_damage_consequence_RV(
#     perf_model_df, cmp_repair_cost_input_df)
# analysis_B['cmp_dmg_quant_df'], analysis_B['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_B['cmp_damage_df'],
#         analysis_B['cmp_damage_consequence_RV_df'],
#         analysis_B['cmp_quant_RV_df'])
# logging.info('\tDetermining component repair cost')
# analysis_B['cmp_cost_RV_df'] = gen_cmp_cost_RV(
#     analysis_B['cmp_dmg_quant_df'])
# analysis_B['cmp_cost_df'] = calc_cmp_cost(
#     analysis_B['cmp_dmg_quant_df'],
#     analysis_B['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_B['cmp_dmg_quant_eco_df'])
# logging.info('\tSummarizing cost')
# analysis_B['total_cost_df'] = calc_total_cost(
#     analysis_B['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_B['cmp_cost_RV_df'])
# logging.info('\tAnalysis B finished')


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# #                       Analysis C                               #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# logging.info('Starting analysis C')
# analysis_C = {}
# logging.info('\tUsing simulated demands from A')
# analysis_C['edp_RV'] = analysis_A['edp_RV']
# logging.info('\tUsing component quantities from B')
# analysis_C['cmp_quant_RV_df'] = analysis_B['cmp_quant_RV_df']
# logging.info('\tUsing component damage state thresholds from B')
# analysis_C['cmp_fragility_RV_df'] = analysis_B['cmp_fragility_RV_df']
# logging.info('\tDetermining component damage')
# analysis_C['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_C['edp_RV'],
#     analysis_C['cmp_fragility_RV_df'])
# logging.info('\tDetermining damage consequences')
# analysis_C['cmp_damage_consequence_RV_df'] = \
#     analysis_B['cmp_damage_consequence_RV_df']
# analysis_C['cmp_dmg_quant_df'], analysis_C['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_C['cmp_damage_df'],
#         analysis_C['cmp_damage_consequence_RV_df'],
#         analysis_C['cmp_quant_RV_df'])
# logging.info('\tDetermining component repair cost')
# analysis_C['cmp_cost_RV_df'] = analysis_B['cmp_cost_RV_df']
# analysis_C['cmp_cost_df'] = calc_cmp_cost(
#     analysis_C['cmp_dmg_quant_df'],
#     analysis_C['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_C['cmp_dmg_quant_eco_df'])
# logging.info('\tSummarizing cost')
# analysis_C['total_cost_df'] = calc_total_cost(
#     analysis_C['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_C['cmp_cost_RV_df'])
# logging.info('\tAnalysis C finished')

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# #                     Sensitivity Indices                        #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# logging.info('Calculating sensitivity indices for the EDPs')
# yA = analysis_A['total_cost_df'].to_numpy()
# yB = analysis_B['total_cost_df'].to_numpy()
# yC = analysis_C['total_cost_df'].to_numpy()


# def calc_sens(yA, yB, yC):
#     n = len(yA)
#     f0 = 1/n * np.sum(yA)
#     s1 = (1./n * (yA.T @ yC) - f0**2) / (1./n * (yA.T @ yA) - f0**2)
#     sT = 1. - (1/n * (yB.T @ yC) - f0**2) / \
#         (1/n * (yA.T @ yA) - f0**2)
#     return s1, sT


# s1, sT = calc_sens(yA, yB, yC)
# print()
# print("EDPs")
# print(f"1st order sensitivity index: {s1:.3f}")
# print(f"Total sensitivity index    : {sT:.3f}")



# num_repeats = 10000
# sel = np.random.choice(num_realizations, (num_repeats, num_realizations))
# bootstrap_sample = np.zeros(num_repeats)
# for j in range(num_repeats):
#     bootstrap_sample[j] = calc_sens(yA[sel[j]], yB[sel[j]], yC[sel[j]])[0]


# from scipy import stats
# plt.hist(bootstrap_sample, 50, density=True, color='lightgrey', edgecolor='black')
# xv = np.linspace(0.60, 0.70, 10000)
# mub = bootstrap_sample.mean()
# mus = bootstrap_sample.std()
# yv = stats.norm.pdf(xv, loc=mub, scale=mus)
# plt.plot(xv, yv)
# plt.show()












# # now some quick check, will go back and fix code organization

# # component fragility uncertainty

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# #                       Analysis D                               #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# logging.info('Starting analysis D')
# analysis_D = {}

# logging.info('\tUsing simulated demands from B')
# analysis_D['edp_RV'] = analysis_B['edp_RV']

# logging.info('\tUsing component quantities from B')
# analysis_D['cmp_quant_RV_df'] = analysis_B['cmp_quant_RV_df']

# logging.info('\tUsing component damage state thresholds from A')
# temp = analysis_A['cmp_fragility_RV_df']
# subset_cols = [('collapse', '0', '1', '0', 'DS1'),
#                ('irreparable', '0', '1', '0', 'DS1')]
# temp.loc[:, subset_cols] = \
#     analysis_B['cmp_fragility_RV_df'].loc[:, subset_cols]
# analysis_D['cmp_fragility_RV_df'] = temp

# logging.info('\tDetermining component damage')
# analysis_D['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_D['edp_RV'],
#     analysis_D['cmp_fragility_RV_df'])

# logging.info('\tDetermining damage consequences')
# analysis_D['cmp_damage_consequence_RV_df'] = \
#     analysis_A['cmp_damage_consequence_RV_df']
# analysis_D['cmp_dmg_quant_df'], analysis_D['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_D['cmp_damage_df'],
#         analysis_D['cmp_damage_consequence_RV_df'],
#         analysis_D['cmp_quant_RV_df'])

# logging.info('\tDetermining component repair cost')
# analysis_D['cmp_cost_RV_df'] = analysis_B['cmp_cost_RV_df']
# analysis_D['cmp_cost_df'] = calc_cmp_cost(
#     analysis_D['cmp_dmg_quant_df'],
#     analysis_D['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_dmg_quant_eco_df'])

# logging.info('\tSummarizing cost')
# analysis_D['total_cost_df'] = calc_total_cost(
#     analysis_D['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_cost_RV_df'])
# logging.info('\tAnalysis D finished')
































# logging.info('Calculating sensitivity indices for the EDPs')
# yA = analysis_A['total_cost_df'].to_numpy()
# yB = analysis_B['total_cost_df'].to_numpy()
# yC = analysis_D['total_cost_df'].to_numpy()


# def calc_sens(yA, yB, yC):
#     n = len(yA)
#     f0 = 1/n * np.sum(yA)
#     s1 = (1./n * (yA.T @ yC) - f0**2) / (1./n * (yA.T @ yA) - f0**2)
#     sT = 1. - (1/n * (yB.T @ yC) - f0**2) / \
#         (1/n * (yA.T @ yA) - f0**2)
#     return s1, sT


# s1, sT = calc_sens(yA, yB, yC)
# print()
# print("EDPs")
# print(f"1st order sensitivity index: {s1:.3f}")
# print(f"Total sensitivity index    : {sT:.3f}")



# num_repeats = 10000
# sel = np.random.choice(num_realizations, (num_repeats, num_realizations))
# bootstrap_sample = np.zeros(num_repeats)
# for j in range(num_repeats):
#     bootstrap_sample[j] = calc_sens(yA[sel[j]], yB[sel[j]], yC[sel[j]])[0]


# from scipy import stats
# plt.hist(bootstrap_sample, 50, density=True, color='lightgrey', edgecolor='black')
# xv = np.linspace(0.60, 0.70, 10000)
# mub = bootstrap_sample.mean()
# mus = bootstrap_sample.std()
# yv = stats.norm.pdf(xv, loc=mub, scale=mus)
# plt.plot(xv, yv)
# plt.show()

















































# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>component cost setup


# logging.info('Starting analysis D')
# analysis_D = {}

# logging.info('\tUsing simulated demands from B')
# analysis_D['edp_RV'] = analysis_B['edp_RV']

# logging.info('\tUsing component quantities from B')
# analysis_D['cmp_quant_RV_df'] = analysis_B['cmp_quant_RV_df']

# logging.info('\tUsing component damage state thresholds from B')
# analysis_D['cmp_fragility_RV_df'] = analysis_B['cmp_fragility_RV_df']

# logging.info('\tDetermining component damage')
# analysis_D['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_D['edp_RV'],
#     analysis_D['cmp_fragility_RV_df'])

# logging.info('\tDetermining damage consequences')
# analysis_D['cmp_damage_consequence_RV_df'] = \
#     analysis_B['cmp_damage_consequence_RV_df']
# analysis_D['cmp_dmg_quant_df'], analysis_D['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_D['cmp_damage_df'],
#         analysis_D['cmp_damage_consequence_RV_df'],
#         analysis_D['cmp_quant_RV_df'])

# logging.info('\tDetermining component repair cost')
# temp = analysis_A['cmp_cost_RV_df']
# temp.loc[:, ('replacement', '0', '1', '0', 'DS1', '1')] = \
#     analysis_B['cmp_cost_RV_df'].loc[:, ('replacement', '0', '1', '0', 'DS1', '1')]
# analysis_D['cmp_cost_RV_df'] = temp
# analysis_D['cmp_cost_df'] = calc_cmp_cost(
#     analysis_D['cmp_dmg_quant_df'],
#     analysis_D['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_dmg_quant_eco_df'])

# logging.info('\tSummarizing cost')
# analysis_D['total_cost_df'] = calc_total_cost(
#     analysis_D['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_cost_RV_df'])
# logging.info('\tAnalysis D finished')

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>component cost setup












# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>fragility setup

# logging.info('Starting analysis D')
# analysis_D = {}

# logging.info('\tUsing simulated demands from B')
# analysis_D['edp_RV'] = analysis_B['edp_RV']

# logging.info('\tUsing component quantities from B')
# analysis_D['cmp_quant_RV_df'] = analysis_B['cmp_quant_RV_df']

# logging.info('\tUsing component damage state thresholds from A')
# temp = analysis_A['cmp_fragility_RV_df']
# subset_cols = [('collapse', '0', '1', '0', 'DS1'),
#                ('irreparable', '0', '1', '0', 'DS1')]
# temp.loc[:, subset_cols] = \
#     analysis_B['cmp_fragility_RV_df'].loc[:, subset_cols]
# analysis_D['cmp_fragility_RV_df'] = temp

# logging.info('\tDetermining component damage')
# analysis_D['cmp_damage_df'] = calc_cmp_damage(
#     perf_model_df, cmp_fragility_input_df,
#     analysis_D['edp_RV'],
#     analysis_D['cmp_fragility_RV_df'])

# logging.info('\tDetermining damage consequences')
# analysis_D['cmp_damage_consequence_RV_df'] = \
#     analysis_A['cmp_damage_consequence_RV_df']
# analysis_D['cmp_dmg_quant_df'], analysis_D['cmp_dmg_quant_eco_df'] = \
#     calc_cmp_dmg_quant(
#         perf_model_df, cmp_repair_cost_input_df,
#         analysis_D['cmp_damage_df'],
#         analysis_D['cmp_damage_consequence_RV_df'],
#         analysis_D['cmp_quant_RV_df'])

# logging.info('\tDetermining component repair cost')
# analysis_D['cmp_cost_RV_df'] = analysis_B['cmp_cost_RV_df']
# analysis_D['cmp_cost_df'] = calc_cmp_cost(
#     analysis_D['cmp_dmg_quant_df'],
#     analysis_D['cmp_cost_RV_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_dmg_quant_eco_df'])

# logging.info('\tSummarizing cost')
# analysis_D['total_cost_df'] = calc_total_cost(
#     analysis_D['cmp_cost_df'],
#     cmp_repair_cost_input_df,
#     analysis_D['cmp_cost_RV_df'])
# logging.info('\tAnalysis D finished')



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>fragility setup



































analysis_A['total_cost_df'].describe()
