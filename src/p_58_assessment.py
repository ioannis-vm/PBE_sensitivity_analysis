"""
Python implementation of the P-58 methodology
for the estimation of the repair cost of buildings
subjected to earthquakes.
Heavily inspired by Pelicun: https://github.com/NHERI-SimCenter/pelicun
This module does not stand as an alternative, but rather aims to ease
the implementation of variance-based sensitivity analysis of the
methodology, suporting merely a subset of  Pelicun's functionality.
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   February 2022
#
# https://github.com/ioannis-vm


from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.stats import norm
import logging

idx = pd.IndexSlice

np.random.seed(42)

# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name


def lognormal_transform(unif_sample, delta, beta):
    return np.exp(norm.ppf(unif_sample, loc=np.log(delta), scale=beta))


def normal_transform(unif_sample, mean, stdev):
    return norm.ppf(unif_sample, loc=mean, scale=stdev)


@dataclass
class P58_Assessment:
    """
    P-58 assessment object.
    A collector of Pandas DataFrames that facilitate
    the implementation of the methodology.

    Methods starting with 'gen_' involve
    random variable sampling.
    Methods starting with 'calc_' involve
    computations that do not involve random
    variable sampling.
    Methods starting with 'read_' involve
    parsing user input (in the form of CSV files).

    num_realizations (int): Number of realizations used.
    replacement_threshold (float): Total cost above which the building
        is deemed irreparable, normalized by the expected replacement
        cost of the building. FEMA P-58 suggests using a
        value of 0.4~0.5.
    fix_epd_mean (bool): Testing ~ fixes edp realizations
                         to the mean.
    fix_epd_mean (bool): Testing ~ fixes component quantity realizations
                         to the mean.
    fix_cmp_dm_mean (bool): Testing ~ fixes component damage threshold
                         realizations to the mean.
    fix_blg_dm_mean (bool): Testing ~ fixes building damage threshold
                            realizations to the mean.
    fix_cmp_dv_mean (bool): Testing ~ fixes component cost
                         realizations to the mean.
    fix_blg_dv_mean (bool): Testing ~ fixes building replacement cost
                            realizations to the mean.
    From user input files:
    perf_model (pd.DataFrame): Performance model
    cmp_fragility_input (pd.DataFrame): Fragility curves
    cmp_repair_cost_input (pd.DataFrame): Repair costs
    Intermediate resutls:
    edp_samples (pd.DataFrame): Contains realizations of
                jointly lognormal building response quantities.
    cmp_quant_RV (pd.DataFrame): Contains realizations of
                component quantity values in their respective units.
    cmp_fragility_RV (pd.DataFrame): Contains realizations of
                component fragility thresholds
    cmp_damage (pd.DataFrame): Contains realizations of
                damage states for each component group
    cmp_damage_consequence_RV (pd.DataFrame): Contains realizations of
                the consequences of any damage state, necessary
                for mutually exclusive damage consequences.
    cmp_dmg_quant (pd.DataFrame): Contains realizations of
                quantities of damaged components.
    cmp_dmg_quant_eco (pd.DataFrame): Contains realizations of
                quantities of damaged components, grouped by
                component only, to account for cost reduction
                due to economies of scale.
    cmp_cost_RV: (pd.DataFrame): Contains realizations of
                U~(0,1) RV's used to obtain repair cost realizations
                via inverce CDF sampling.
    cmp_cost: (pd.DataFrame): Contains realizations of
                component repair costs
    Final resutls:
    total_cost: (pd.DataFrame): Contains realizations of
                the total repair cost.
    """

    logFile: str = field(default=None)
    num_realizations: int = field(default=1000)
    replacement_threshold: float = field(default=1.00)
    fix_epd_mean: bool = field(default=False)
    fix_quant_mean: bool = field(default=False)
    fix_cmp_dm_mean: bool = field(default=False)
    fix_blg_dm_mean: bool = field(default=False)
    fix_cmp_dv_mean: bool = field(default=False)
    fix_blg_dv_mean: bool = field(default=False)
    perf_model: pd.DataFrame = field(init=False, repr=False)
    cmp_fragility_input: pd.DataFrame = field(init=False, repr=False)
    cmp_repair_cost_input: pd.DataFrame = field(init=False, repr=False)
    edp_samples: pd.DataFrame = field(init=False, repr=False)
    cmp_quant_RV: pd.DataFrame = field(init=False, repr=False)
    cmp_fragility_RV: pd.DataFrame = field(init=False, repr=False)
    cmp_damage: pd.DataFrame = field(init=False, repr=False)
    cmp_damage_consequence_RV: pd.DataFrame = field(init=False, repr=False)
    cmp_dmg_quant: pd.DataFrame = field(init=False, repr=False)
    cmp_dmg_quant_eco: pd.DataFrame = field(init=False, repr=False)
    cmp_cost_RV: pd.DataFrame = field(init=False, repr=False)
    cmp_cost: pd.DataFrame = field(init=False, repr=False)
    total_cost: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        logging.basicConfig(
            filename=self.logFile,
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.DEBUG)

    def gen_edp_samples(self, resp_path, c_mdl):
        logging.info('\tGenerating simulated demands')
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
            mean_vec, covariance_mat,
            size=self.num_realizations)
        samples_raw = np.exp(z_mat)
        samples = pd.DataFrame(samples_raw, index=None)
        samples.columns = data.columns
        samples.index.name = 'realization'
        samples.sort_index(axis=1, inplace=True)
        if self.fix_epd_mean:
            samples.loc[:, :] = samples.mean(axis=0).to_numpy()
        self.edp_samples = samples

    def read_perf_model(self, perf_model_input_path):
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
        self.perf_model = perf_model

    def gen_cmp_quant_RV(self):
        logging.info('\tSampling component quantities')
        # initialize empty dataframe
        cmp_quant_RV = pd.DataFrame(
            np.full((self.num_realizations, self.perf_model.shape[0]), 0.00),
            columns=self.perf_model.index, index=None)
        cmp_quant_RV.columns.names = [
            'component', 'location', 'direction', 'group']
        cmp_quant_RV.index.name = 'realizations'
        cmp_quant_RV.sort_index(axis=1, inplace=True)

        # filter components with fixed quantities
        no_distr_idx = self.perf_model[
            self.perf_model['distribution'].isnull()].index
        for comp_group in no_distr_idx:
            cmp_quant_RV.loc[:, comp_group] = \
                self.perf_model.loc[comp_group, 'mean']

        # filter components with lognormal quantities
        lognormal_idx = self.perf_model[self.perf_model['distribution'] ==
                                        'lognormal'].index
        assert(len(lognormal_idx) + len(no_distr_idx) ==
               self.perf_model.shape[0]), \
            "Only lognormal is supported here."
        for comp_group in lognormal_idx:
            mu = self.perf_model.loc[comp_group, 'mean']
            cov = self.perf_model.loc[comp_group, 'cov']
            sgm = mu * cov
            mu_n = np.log(mu) - 0.50 * np.log(1. + (sgm/mu)**2)
            sgm_n = np.sqrt(np.log(1. + (sgm/mu)**2))
            vec = np.exp(np.random.normal(mu_n, sgm_n, self.num_realizations))
            cmp_quant_RV.loc[:, comp_group] = vec
        if self.fix_quant_mean:
            cmp_quant_RV.loc[:, :] = cmp_quant_RV.mean(axis=0).to_numpy()
        self.cmp_quant_RV = cmp_quant_RV

    def read_fragility_input(self, cmp_fragility_input_path):
        cmp_fragility_input = pd.read_csv(
            cmp_fragility_input_path, index_col=0)
        cmp_fragility_input.sort_index(inplace=True)
        self.cmp_fragility_input = cmp_fragility_input

    def gen_cmp_fragility_RV(self):
        logging.info('\tSampling component damage state thresholds')
        # instantiate a dataframe with the right indices
        all_groups = self.perf_model.index.values
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
                delta = self.cmp_fragility_input.loc[comp_id, f'DS{i+1}_delta']
                if pd.isna(delta):
                    continue
                col_idx.extend([(*item, f'DS{i+1}')
                                for item in list_of_these_groups])
        cmp_fragility_RV = pd.DataFrame(
            index=range(self.num_realizations),
            columns=pd.MultiIndex.from_tuples(col_idx))
        cmp_fragility_RV.columns.names = ['component', 'location', 'direction',
                                          'group', 'damage state']
        cmp_fragility_RV.sort_index(axis=1, inplace=True)
        cmp_fragility_RV.sort_index(axis=0, inplace=True)
        # sample damage state threshold values
        for group in all_groups:
            comp_id = group[0]
            unif_sample = np.random.uniform(0.00, 1.00, self.num_realizations)
            for i in range(3):
                delta = self.cmp_fragility_input.loc[comp_id, f'DS{i+1}_delta']
                if pd.isna(delta):
                    continue
                beta = self.cmp_fragility_input.loc[comp_id, f'DS{i+1}_beta']
                sample = lognormal_transform(unif_sample, delta, beta)
                sample *= (delta / np.median(sample))
                cmp_fragility_RV.loc[:, (*group, f'DS{i+1}')] = sample

        if self.fix_cmp_dm_mean:
            cols = cmp_fragility_RV.columns
            cols = cols[:-2]  # remove 'collapse, 'irreparable'
            cmp_fragility_RV.loc[:, cols] = \
                cmp_fragility_RV.loc[:, cols].mean(axis=0).to_numpy()
        if self.fix_blg_dm_mean:
            cols = [('collapse', '0', '1', '0', 'DS1'),
                    ('irreparable', '0', '1', '0', 'DS1')]
            cmp_fragility_RV.loc[:, cols] = \
                cmp_fragility_RV.loc[:, cols].mean(axis=0).to_numpy()
            
        self.cmp_fragility_RV = cmp_fragility_RV

    def calc_cmp_damage(self):
        logging.info('\tDetermining component damage')
        all_groups = self.perf_model.index.values
        cmp_damage = pd.DataFrame(
            np.full((self.num_realizations,
                     len(all_groups)), 0, dtype=np.int32),
            index=range(self.num_realizations),
            columns=pd.MultiIndex.from_tuples(all_groups))
        cmp_damage.columns.names = \
            ['component', 'location', 'direction', 'group']
        cmp_damage.sort_index(axis=1, inplace=True)
        cmp_damage.sort_index(axis=0, inplace=True)
        for group in all_groups:
            fragility_info = self.cmp_fragility_input.loc[group[0], :]
            if fragility_info.directional == 1:
                edp_vec = self.edp_samples.loc[
                    :, (fragility_info['edp type'], group[1], group[2])]
            else:
                edp_vec = self.edp_samples.loc[
                    :, (fragility_info['edp type'], group[1])].max(
                        axis=1) * 1.2
            for i in range(3):
                delta = self.cmp_fragility_input.loc[
                    group[0], f'DS{i+1}_delta']
                if pd.isna(delta):
                    continue
                ds_mask = edp_vec > self.cmp_fragility_RV.loc[
                    :, (*group, f'DS{i+1}')]
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
            cmp_damage.loc[
                indx, ('C3021.001k', str(int(loc)-1), drct, group)] = 1

        # when D4011.021a reaches DS2 --> C3021.001k DS1 is enabled
        cols = cmp_damage.loc[:, ('D4011.021a')].columns
        for col in cols:
            loc, drct, group = col
            sub = cmp_damage.loc[:, ('D2021.011a', *col)]
            indx = sub[sub == 2].index
            cmp_damage.loc[
                indx, ('C3021.001k', str(int(loc)-1), drct, group)] = 1
        self.cmp_damage = cmp_damage

    def read_cmp_repair_cost_input(self, cmp_repair_cost_input_path):
        cmp_repair_cost_input = pd.read_csv(
            cmp_repair_cost_input_path, index_col=0)
        cmp_repair_cost_input.drop(
            ['description', 'note'], inplace=True, axis=1)

        def pipe_split(x, side):
            if pd.isna(x):
                res = pd.NA
            else:
                if '|' in x:
                    res = x.split('|')[side]
                    if ',' in res:
                        res = res.split(',')
                        res = [float(x) for x in res]
                    else:
                        res = [float(res), float(res)]
                else:
                    if side == 0:
                        res = [float(x), float(x)]
                    else:
                        res = [2., 1.]
            return res

        damage_consequences = ['DS1_1', 'DS1_2', 'DS2_1', 'DS3_1']
        for dc in damage_consequences:
            cmp_repair_cost_input[f'{dc}_theta0_c'] = \
                cmp_repair_cost_input[f'{dc}_theta0'].apply(pipe_split, side=0)
            cmp_repair_cost_input[f'{dc}_theta0_q'] = \
                cmp_repair_cost_input[f'{dc}_theta0'].apply(pipe_split, side=1)
            cmp_repair_cost_input.drop(f'{dc}_theta0', inplace=True, axis=1)
        self.cmp_repair_cost_input = cmp_repair_cost_input

    def num_DS(self, comp_id):
        nds = 0
        for i in range(3):
            if not np.isnan(
                    self.cmp_repair_cost_input.loc[comp_id][f'DS{i+1}_n']):
                nds += 1
        return nds

    def gen_cmp_damage_consequence_RV(self):
        logging.info('\tDetermining damage consequences')
        # pick damage state consequence
        # generate columns
        cols = []
        all_groups = self.perf_model.index.values
        for group in all_groups:
            comp_id = group[0]
            if comp_id in ['collapse', 'irreparable']:
                continue
            nds = self.num_DS(comp_id)
            for i_s in range(nds):
                cols.append((*group, f'DS{i_s+1}'))
        cols.append(('replacement', '0', '1', '0', 'DS1'))
        col_idx = pd.MultiIndex.from_tuples(cols)
        col_idx.names = ['component', 'location', 'direction',
                         'group', 'damage state']
        cmp_damage_consequence_RV = pd.DataFrame(
            columns=col_idx, index=range(self.num_realizations))
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
            w1 = self.cmp_repair_cost_input.loc[comp_id, :]['DS1_1_w']
            w2 = self.cmp_repair_cost_input.loc[comp_id, :]['DS1_2_w']
            if not np.isnan(w2):
                assert(w1 + w2 - 1.00 < 1e-5)
                n_cols = cmp_damage_consequence_RV.loc[
                    :, idx[comp_id, :, :, :, 'DS1']].shape[1]
                mat = np.random.binomial(
                    1, w2, (self.num_realizations, n_cols)) + 1
                cmp_damage_consequence_RV.loc[
                    :, idx[comp_id, :, :, :, 'DS1']] = mat
            else:
                cmp_damage_consequence_RV.loc[
                    :, idx[comp_id, :, :, :, 'DS1']] = 1
        self.cmp_damage_consequence_RV = cmp_damage_consequence_RV

    def calc_cmp_dmg_quant(self):
        # generate columns
        cols = []
        # ds_len = [2, 1, 1]
        all_groups = self.perf_model.index.values
        for group in all_groups:
            comp_id = group[0]
            if comp_id in ['collapse', 'irreparable']:
                continue
            nds = self.num_DS(comp_id)
            for i_s in range(nds):
                n_cons = int(self.cmp_repair_cost_input.loc[
                    comp_id][f'DS{i_s+1}_n'])
                for consequence in range(n_cons):
                    cols.append((*group, f'DS{i_s+1}', str(consequence + 1)))
        cols.append(('replacement', '0', '1', '0', 'DS1', '1'))
        col_idx = pd.MultiIndex.from_tuples(cols)
        col_idx.names = ['component', 'location', 'direction',
                         'group', 'damage state', 'consequence']
        # gather damaged quantities
        cmp_dmg_quant = pd.DataFrame(
            np.zeros((self.num_realizations, len(col_idx))),
            columns=col_idx, index=range(self.num_realizations))
        cmp_dmg_quant.index.name = 'realizations'
        cmp_dmg_quant.sort_index(axis=1, inplace=True)
        for i_col, col in enumerate(self.cmp_damage.columns):
            if col[0] in ['collapse', 'irreparable']:
                continue
            vec = self.cmp_damage.iloc[:, i_col]
            nds = self.num_DS(col[0])
            n_cons = int(self.cmp_repair_cost_input.loc[
                col[0]]['DS1_n'])
            idx_ds1 = vec[vec == 1].index
            idx_ds1_1 = self.cmp_damage_consequence_RV.loc[
                idx_ds1, (*col, 'DS1')][
                self.cmp_damage_consequence_RV.loc[
                    idx_ds1, (*col, 'DS1')] == 1].index
            cmp_dmg_quant.loc[idx_ds1_1, (*col, 'DS1', '1')] = \
                self.cmp_quant_RV.loc[idx_ds1_1, col]
            if n_cons > 1:
                idx_ds1_2 = self.cmp_damage_consequence_RV.loc[
                    idx_ds1, (*col, 'DS1')][
                    self.cmp_damage_consequence_RV.loc[
                        idx_ds1, (*col, 'DS1')] == 2].index
                cmp_dmg_quant.loc[idx_ds1_2, (*col, 'DS1', '2')] = \
                    self.cmp_quant_RV.loc[idx_ds1_2, col]
            if nds > 1:
                idx_ds2 = vec[vec == 2].index
                cmp_dmg_quant.loc[idx_ds2, (*col, 'DS2', '1')] = \
                    self.cmp_quant_RV.loc[idx_ds2, col]
            if nds > 2:
                idx_ds3 = vec[vec == 3].index
                cmp_dmg_quant.loc[idx_ds3, (*col, 'DS3', '1')] = \
                    self.cmp_quant_RV.loc[idx_ds3, col]
        for i_col, col in enumerate(self.cmp_damage.columns):
            if col[0] in ['collapse', 'irreparable']:
                vec = self.cmp_damage.iloc[:, i_col]
                indx = vec[vec == 1].index
                cmp_dmg_quant.loc[indx, :] = 0.00
                cmp_dmg_quant.loc[
                    indx, ('replacement', '0', '1', '0', 'DS1', '1')] = \
                    self.cmp_quant_RV.loc[indx, col]
        # assuming economy of scale, we add the damaged quantities
        # across location, direction, and group.
        cmp_dmg_quant_eco = cmp_dmg_quant.groupby(
            level=['component'], axis=1).sum()
        self.cmp_dmg_quant = cmp_dmg_quant
        self.cmp_dmg_quant_eco = cmp_dmg_quant_eco

    def gen_cmp_cost_RV(self):
        logging.info('\tSampling component repair cost random variables')
        cmp_cost_RV = pd.DataFrame(
            np.random.uniform(
                0.00, 1.00,
                (self.num_realizations, len(self.cmp_dmg_quant.columns))),
            columns=self.cmp_dmg_quant.columns,
            index=range(self.num_realizations))
        cmp_cost_RV.index.name = 'realizations'
        cmp_cost_RV.sort_index(axis=1, inplace=True)
        self.cmp_cost_RV = cmp_cost_RV

    def calc_cmp_cost(self):
        logging.info('\tCalculating component repair cost')
        cmp_cost = pd.DataFrame(
            np.zeros((self.num_realizations, len(self.cmp_dmg_quant.columns))),
            columns=self.cmp_dmg_quant.columns,
            index=self.cmp_cost_RV.index)
        cmp_cost.index.name = 'realizations'
        cmp_cost.sort_index(axis=1, inplace=True)
        # (this could be made more efficient if it was looping over
        #  components instead of looping over groups)
        # but the result is the same
        for i_col, col in enumerate(self.cmp_cost_RV.columns):
            comp_id, location, direction, group, \
                damage_state, consequence = col
            b_q = self.cmp_repair_cost_input.loc[comp_id, :]['base_quantity']
            median_dmg_quant = self.cmp_dmg_quant_eco.loc[:, comp_id] / b_q
            es_x = self.cmp_repair_cost_input.loc[
                comp_id, :][f'{damage_state}_{consequence}_theta0_q']
            es_y = self.cmp_repair_cost_input.loc[
                comp_id, :][f'{damage_state}_{consequence}_theta0_c']
            cost = np.zeros(self.num_realizations)
            sel = median_dmg_quant < es_x[0]
            cost[sel] = es_y[0]
            sel = median_dmg_quant > es_x[1]
            cost[sel] = es_y[1]
            sel = np.logical_and(median_dmg_quant <= es_x[1],
                                 median_dmg_quant >= es_x[0])
            cost[sel] = (es_y[1]-es_y[0])/(es_x[1]-es_x[0])*(
                median_dmg_quant[sel]-es_x[0])+es_y[0]
            # add variability
            distribution = self.cmp_repair_cost_input.loc[comp_id][
                f'{damage_state}_{consequence}_distribution']
            if distribution == 'normal':
                cov = self.cmp_repair_cost_input.loc[comp_id, :][
                    f'{damage_state}_{consequence}_theta1']
                sample = normal_transform(
                    self.cmp_cost_RV.loc[:, col], 1.00, cov)
                sample /= np.mean(sample)
                cmp_cost.loc[:, col] = self.cmp_dmg_quant.loc[
                    :, col] / b_q * cost * sample
                if comp_id != 'replacement' and self.fix_cmp_dv_mean:
                    cmp_cost.loc[:, col] = self.cmp_dmg_quant.loc[
                        :, col] / b_q * cost
                if comp_id == 'replacement' and self.fix_blg_dv_mean:
                    cmp_cost.loc[:, col] = self.cmp_dmg_quant.loc[
                        :, col] / b_q * cost

            elif distribution == 'lognormal':
                beta = self.cmp_repair_cost_input.loc[comp_id, :][
                    f'{damage_state}_{consequence}_theta1']
                sample = lognormal_transform(
                    self.cmp_cost_RV.loc[:, col], 1.00, beta)
                sample /= np.median(sample)
                cmp_cost.loc[:, col] = self.cmp_dmg_quant.loc[
                    :, col] / b_q * cost * sample
                if comp_id != 'replacement' and self.fix_cmp_dv_mean:
                    cmp_cost.loc[:, col] = self.cmp_dmg_quant.loc[
                        :, col] / b_q * cost * np.exp(beta**2/2)
                if comp_id == 'replacement' and self.fix_blg_dv_mean:
                    cmp_cost.loc[:, col] = self.cmp_dmg_quant.loc[
                        :, col] / b_q * cost * np.exp(beta**2/2)
            elif distribution == 'zero':
                cmp_cost.loc[:, col] = 0.00
            else:
                raise ValueError(
                    f'Unknown distribution encountered: {distribution}')
        self.cmp_cost = cmp_cost

    def calc_total_cost(self):
        logging.info('\tSummarizing cost')
        total_cost = self.cmp_cost.sum(axis=1)
        # consider replacement cost threshold value
        col = ('replacement', '0', '1', '0', 'DS1', '1')
        rpl = self.cmp_cost.loc[:, col]
        non_replacement_idx = rpl[rpl == 0.00].index
        n_samples = len(non_replacement_idx)
        if n_samples != 0:
            delta = self.cmp_repair_cost_input.loc[
                'replacement', 'DS1_1_theta0_c'][0]
            beta = self.cmp_repair_cost_input.loc[
                'replacement', 'DS1_1_theta1']
            mean = delta * np.exp(beta**2/2.)
            change_idx = total_cost.loc[
                non_replacement_idx][
                    total_cost.loc[
                        non_replacement_idx] >
                    self.replacement_threshold * mean].index
            sample = lognormal_transform(
                self.cmp_cost_RV.loc[change_idx, col], delta, beta)
            sample *= delta/np.median(sample)
            total_cost[change_idx] = sample
            if self.fix_blg_dv_mean:
                total_cost[change_idx] = mean
        self.total_cost = total_cost

    def run(self, response_path, c_mdl):
        self.gen_edp_samples(response_path, c_mdl)
        self.gen_cmp_quant_RV()
        self.gen_cmp_fragility_RV()
        self.gen_cmp_damage_consequence_RV()
        self.calc_cmp_damage()
        self.calc_cmp_dmg_quant()
        self.gen_cmp_cost_RV()
        self.calc_cmp_cost()
        self.calc_total_cost()


def calc_sens(yA: np.ndarray,
              yB: np.ndarray,
              yC: np.ndarray,
              yD: np.ndarray) -> tuple[float, float]:
    """
    Calculate variance-based 1st-order and total effect sensitivity
    indices based on the procedure outlined in Saltelli (2002) and
    subsequent improvements discussed in Yun et al. (2017).

    - Saltelli, Andrea. "Making best use of model evaluations to
    compute sensitivity indices." Computer physics communications
    145.2 (2002): 280-297.
    - Yun, Wanying, et al. "An efficient sampling method for
    variance-based sensitivity analysis." Structural Safety 65 (2017):
    74-83.

    Args:
    yA (np.ndarray): One-dimensional numpy array containing realizations
       of model evaluations of analysis 'A'.
    yB (np.ndarray): One-dimensional numpy array containing realizations
       of model evaluations of analysis 'B'
       (every random variable resampled).
    yC (np.ndarray): One-dimensional numpy array containing realizations
       of model evaluations of analysis 'C'
       (reusing all input realizations of B, except for a single one where
        those of A are used).
    yD (np.ndarray): One-dimensional numpy array containing realizations
       of model evaluations of analysis 'D'
       (reusing all input realizations of A, except for a single one where
        those of B are used).
    Returns:
    s1, sT: First-order and total effect sensitivity indices
    """

    # simplest method
    # n = len(yA)
    # f0 = 1./n * np.sum(yA)
    # s1 = ((1./n)*(np.dot(yA, yC)) - f0**2)\
    #     / ((1./n)*(np.dot(yA, yA))-f0**2)
    # sT = 1. - ((1./n)*np.dot(yB, yC) - f0**2)\
    #     / ((1./n)*np.dot(yA, yA) - f0**2)

    # full-use method
    n = len(yA)
    f0_sq = 1./(2.*n) * np.sum(yA * yB + yC * yD)
    s1 = ((1/(2.*n)) * (np.sum(yA * yC) + np.sum(yB * yD)) - f0_sq) / ((1./(2.*n)) * (np.sum(yA**2 + yB**2)) - f0_sq)
    sT = 1 - ((1/(2.*n)) * (np.sum(yB * yC) + np.sum(yA * yD)) - f0_sq) / ((1./(2.*n)) * (np.sum(yA**2 + yB**2)) - f0_sq)

    # # Jansen's method
    # n = len(yA)
    # s1 = (1. - ((1./n) * np.sum((yB-yD)**2)) /
    #       ((1./n) * np.sum(yB**2 + yD**2) -
    #        ((1./n) * np.sum(yB))**2 -
    #        ((1./n) * np.sum(yD))**2))
    # sT = (((1./n) * np.sum((yA-yD)**2)) /
    #       ((1./n) * np.sum(yA**2 + yD**2) -
    #        ((1./n) * np.sum(yA))**2 -
    #        ((1./n) * np.sum(yD))**2))

    return s1, sT
