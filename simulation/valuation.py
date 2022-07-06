import pandas as pd
import numpy as np
from typing import Callable, Dict, Union
import numpy_financial as npf


class Valuation:
    def __init__(self, data: pd.DataFrame, valuation_period: int, deal_volume: int,
                 single_investment_size: Union[int, float], incubation_success_ratio: float,
                 opex: Union[int, float], varex: Union[int, float]):
        self.data = data
        self.valuation_period = valuation_period
        self.single_investment_size = single_investment_size
        self.deal_volume = deal_volume
        self.successes = deal_volume * valuation_period * incubation_success_ratio
        self.investment_size = single_investment_size
        self._total_invested = deal_volume * single_investment_size * valuation_period
        self.result: Dict[str, Dict[str, float]] = {}
        self._fee_valuation = 0
        self._equity_valuation = 0
        self._total_valuation = 0
        self._inflation = 0.05
        self._net_valuation = 0
        self._cagr = 0
        self._coc = 0
        self.opex_annual = opex
        self.varex_per_deal = varex
        self._opex = opex * ((pow(1 + self._inflation, valuation_period)) - 1) / self._inflation
        self._varex = varex * deal_volume * valuation_period
        self._total_expenses = self._opex + self._varex

    def pivot(self, agg_field: str, agg_func: Callable = np.sum) -> pd.DataFrame:
        """
        Use np.mean when agg_field is `CAGR`; else use np.sum
        """
        pivot_data = self.data.pivot_table(index='sim_no', columns='year', values=agg_field, aggfunc=agg_func,
                                           fill_value=0).reset_index()
        year_start_pivot_data = self.data.pivot_table(index='sim_no', values='year_invested', aggfunc=np.mean,
                                                      fill_value=0).reset_index()
        pivot_data = pd.concat([pivot_data, year_start_pivot_data], axis=1).T.drop_duplicates().T
        return pivot_data

    def split_quantiles(self, pivot_data: pd.DataFrame, quantile_increment_count: int = 5) -> Dict[str, float]:
        q = 0
        quantile_increments = 1 / quantile_increment_count
        temp_datasets = {}
        while q < 1:
            q_lower = pivot_data[self.valuation_period].quantile(round(q, 2))
            q += quantile_increments
            q_upper = pivot_data[self.valuation_period].quantile(round(q, 2))
            subset = pivot_data[
                (pivot_data[self.valuation_period] >= q_lower) & (pivot_data[self.valuation_period] < q_upper)]
            temp_datasets[f'q{q}'] = subset
        return temp_datasets

    def run_valuation(self, agg_field: str, agg_func: Callable, ebidta_multiple: int, discount_rate: float):
        coc_multiples = [2, 5, 10, 15, 20]
        group_dist_allocation = [0.2, 0.25, 0.25, 0.2, 0.1]
        pivot_data = self.pivot(agg_field, agg_func)
        quantile_data: Dict[str, float] = self.split_quantiles(pivot_data, 5)
        for i, (key, table) in enumerate(quantile_data.items()):
            year_mean = table['year_invested'].mean()
            management_fee_mean = table[self.valuation_period].mean()
            self.result[key] = {'management_fee_avg': management_fee_mean,
                                'years_discounted': year_mean,
                                'investments_allocated': self.successes * group_dist_allocation[i],
                                'man_fee_valuation': (self.successes * group_dist_allocation[
                                    i]) * management_fee_mean * ebidta_multiple,
                                'equity_valuation': (self.investment_size * (
                                            self.successes * group_dist_allocation[i])) * coc_multiples[i] * pow(
                                    (1 + discount_rate), - year_mean)}
        self.compile_valuation_results()
        return self

    def compile_valuation_results(self) -> None:
        if not self.result:
            "No valuation values have been calculated yet."
        for _, y in self.result.items():
            self._fee_valuation += y['man_fee_valuation']
            self._equity_valuation += y['equity_valuation']
        self._total_valuation = self._fee_valuation + self._equity_valuation

    @property
    def total_invested(self):
        return self._total_invested

    @property
    def equity_valuation(self):
        return self._equity_valuation

    @property
    def fee_valuation(self):
        return self._fee_valuation

    @property
    def total_valuation(self):
        return self._total_valuation

    @property
    def net_valuation(self):
        self._net_valuation = self._total_valuation - self._total_expenses - self._total_invested
        return self._net_valuation

    @property
    def cagr(self):
        if self._total_valuation == 0 or self._net_valuation == 0:
            return "No `net valuation` value has been calculated yet."
        self._cagr = pow((self._net_valuation / self._total_invested), (1 / self.valuation_period)) - 1
        return self._cagr

    @property
    def coc(self):
        if self._total_valuation == 0 or self._net_valuation == 0:
            return "No net valuation value has been calculated yet."
        self._coc = int(round(self._net_valuation / (self._total_invested - self.opex - self.varex), 0))
        return self._coc

    @property
    def irr(self):
        if self._net_valuation == 0:
            return "No valuation result has been created yet."
        cashflows = []
        for x in range(self.valuation_period):
            varex = self.varex_per_deal * self.deal_volume
            opex = self.opex_annual * pow(1.05, x)
            if x == (self.valuation_period - 1):
                cf = self.net_valuation - (self.single_investment_size * self.deal_volume) - (opex + varex)
            else:
                cf = -(self.single_investment_size * self.deal_volume) - (varex + opex)
            cashflows.append(cf)
        return npf.irr(cashflows)

    @property
    def opex(self):
        return self._opex

    @property
    def varex(self):
        return self._varex

    @property
    def total_expenses(self):
        return self._total_expenses

