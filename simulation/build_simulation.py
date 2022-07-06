from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from jinja2 import Template
import random
from typing import Dict, List
import time

# from matplotlib.pyplot import fill
# import numpy_financial as npf

pd.set_option('display.float_format', lambda x: '{:,.4f}'.format(x))


class Db(object):
    def __init__(self, host, port, pword, dbname, user):
        self.host = host
        self.port = port
        self.pword = pword
        self.dbname = dbname
        self.user = user
        self.conn = None

    def __connect(self):
        self.engine = create_engine(f"postgresql://{self.user}:{self.pword}@{self.host}:{self.port}/{self.dbname}")

    def select(self, query):
        self.__connect()
        data: pd.DataFrame = pd.read_sql(query, con=self.engine)
        self.engine.dispose()
        return data


# Create database connection instance
db = Db(host='localhost', port=7432, pword='', dbname='marcobuissinne', user='marcobuissinne')

# SQL cte statement that's used as the base query
BASE_QUERY: str = Template("""
                    --sql            
                    with a as  (
                        SELECT a.branch_no, yyyy, mm, amt, business_sub_type_code, cell_name, business_type
                        from insurance_cells a
                        join insurance_cell_mapping icm on a.branch_no = icm.branch_no
                        where description = 'Management Fees' and business_sub_type_code = '3rd'
                    ),
                    b as (SELECT cell_name,
                                branch_no,
                                yyyy,
                                count(1),
                                row_number() over (partition by branch_no order by yyyy) as annual_growth_period,
                                sum(amt) as fees_standard,
                                sum(amt)*(12/count(1)::numeric) as fees_normalised
                    FROM a
                    group by 1,2,3 order by 2,3
                    ),
                    c as (SELECT *, lag(fees_normalised) over (partition by branch_no order by yyyy) as fees_lagged
                        FROM b where annual_growth_period in (1,2,3,4,5,6,7,8,9,10,11)
                    ),
                    d as (SELECT *, ((fees_normalised / nullif(fees_lagged,0)) - 1) as growth_factor FROM c where fees_lagged is not null)
                    {{ query }};
                    """)


def calculate_growth_percentiles(query, db_connection: Db):
    query_interpolation = BASE_QUERY.render(query=query)
    data = db_connection.select(query_interpolation)
    return data


class Simulate:
    def __init__(self, data_fees: pd.DataFrame, data_growth: pd.DataFrame, valuation_period: int):
        self.data_fees = data_fees
        self.data_growth = data_growth
        self.single_management_fees = []
        self.single_growth_factor = []
        self.single_investment_year = []
        self.single_inv_start_year = []
        self.single_prev_fee = []
        self.single_init_fee = []
        self.single_terminal_fee = []
        self.single_year_count = []
        self.sim_number = []
        self.single_year_zero_based = []
        self._results = None
        self.valuation_period = valuation_period

    # def __repr__(self):
    #     if self.return_data is None:
    #         return "No Simulation has been run yet!"
    #     else:
    #         return self.return_data

    def run(self, sims: int, t):
        if self._results is not None:
            self.__init__(self.data_fees, self.data_growth)
        # self.valuation_period = valuation_period
        self.sims = sims
        self.investment_year_dist: List[Dict[float, int]] = [
            {round(1 / (self.valuation_period) + x * 1 / (self.valuation_period), 2): x + 2} for x in
            range(int(self.valuation_period))]

        for j in range(self.sims, self.sims + t, 1):
            man_fee_rand_number = random.random()
            intial_management_fee = self.data_fees['fees_normalised'].quantile(man_fee_rand_number)
            year_rand_number = random.random()
            for counter, item in enumerate(self.investment_year_dist):
                cum_prob = list(item.keys())[0]
                if year_rand_number <= cum_prob:
                    investment_starting_year = self.investment_year_dist[counter][cum_prob]
                    break
            year_counter = 0
            for i in range(investment_starting_year, self.valuation_period + 2):
                year_counter += 1
                rand_num = random.random()
                fees_growth_factor = self.data_growth.query('annual_growth_period == @i')['growth_factor'].quantile(
                    rand_num)
                self.single_prev_fee.append(intial_management_fee)
                if i == investment_starting_year:
                    [self.single_init_fee.append(intial_management_fee) for _ in
                     range(investment_starting_year, self.valuation_period + 2)]
                intial_management_fee *= (1 + fees_growth_factor)
                # terminal_management_fee = intial_management_fee
                self.single_management_fees.append(intial_management_fee)
                self.single_growth_factor.append(fees_growth_factor)
                self.single_investment_year.append((i - 1))
                self.sim_number.append(j + 1)
                # self.single_init_fee.append(intial_management_fee)
                if i == self.valuation_period + 1:
                    [self.single_terminal_fee.append(intial_management_fee) for _ in
                     range(investment_starting_year, self.valuation_period + 2)]
                    [self.single_year_count.append(year_counter) for _ in
                     range(investment_starting_year, self.valuation_period + 2)]
                self.single_inv_start_year.append(investment_starting_year - 1)
                self.single_year_zero_based.append(year_counter)
        self.results
        return self

    @property
    def results(self):
        self._results = pd.DataFrame({'sim_no': self.sim_number,
                                      'year n - 1 fee': self.single_prev_fee,
                                      'init_man_fee': self.single_init_fee,
                                      'year_invested': self.single_inv_start_year,
                                      'growth_factor': self.single_growth_factor,
                                      'management_fee': self.single_management_fees,
                                      'terminal_management_fee': self.single_terminal_fee,
                                      'year': self.single_investment_year,
                                      'total_years_invested': self.single_year_count,
                                      'current_year': self.single_year_zero_based})
        return self._results


from multiprocessing import Pool


def rename_function(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


def q_at(y):
    @rename_function(f'q{y:0.2f}')
    def q(x):
        return x.quantile(y)

    return q


class Percentile:
    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.data = None
        self.field = None
        self._quantiles = None

    def _filter(self, filter_field: str) -> pd.DataFrame:
        filtered: pd.DataFrame = self.raw_data[self.raw_data[filter_field] == self.raw_data[filter_field].min()]
        return filtered

    @property
    def quantiles(self):
        if self.field is None:
            return "apply_quantile_filters method needs to be run first!"
        quantile_functions = {self.field: [q_at(i) for i in np.arange(0, 1, 0.05)]}
        self._quantiles = self.raw_data.groupby("annual_growth_period").agg(quantile_functions)
        return self._quantiles

    def apply_quantile_filters(self, quantile_field: str, is_fees: bool, lower: float = 0.1,
                               upper: float = 0.9) -> pd.DataFrame:
        self.field = quantile_field
        if is_fees:
            print("Is fees")
            self.raw_data = self._filter("annual_growth_period")
            cutoff_lower = self.raw_data[quantile_field].quantile(lower)
            cutoff_upper = self.raw_data[quantile_field].quantile(upper)
            self.data = self.raw_data[
                (self.raw_data[quantile_field] >= cutoff_lower) & (self.raw_data[quantile_field] <= cutoff_upper)]
        else:
            temp = pd.concat([pd.DataFrame(self.quantiles.iloc[:, int(lower / 0.05)]),
                              pd.DataFrame(self.quantiles.iloc[:, int(upper / 0.05)])], axis=1).reset_index()
            quantiles_join = pd.DataFrame({'growth_period': temp.iloc[:, 0].tolist(),
                                           'lower': temp.iloc[:, 1].tolist(),
                                           'upper': temp.iloc[:, 2].tolist()})
            merged = pd.merge(self.raw_data, quantiles_join, how='left', left_on='annual_growth_period',
                              right_on='growth_period')
            merged.drop(columns=['growth_period'], inplace=True)
            print(quantiles_join)
            self.data = merged[
                (merged['growth_factor'] >= merged['lower']) & (merged['growth_factor'] <= merged['upper'])]
        return self.data


# v = Percentile(summary_data)
# b = v.apply_quantile_filters('growth_factor', False, 0.4, 0.75)
# z = pd.merge(b, c, how='left', left_on='annual_growth_period', right_on='growth_period')
# z.drop(columns=['growth_period'], inplace=True)
# final = z[(z['growth_factor']>= z['lower']) & (z['growth_factor'] <= z['upper'])]
# len(final)


def get_filtered_data(var_field: str, is_fees: bool, lower_bound: float, upper_bound: float) -> Percentile:
    instance = Percentile(summary_data)
    instance.apply_quantile_filters(var_field, is_fees, lower_bound, upper_bound)
    return instance


def simulate_with_pool(sims, valuation_period: int) -> List[Simulate]:
    # get growth factor data for sim
    data_growth_obj = get_filtered_data('growth_factor', False, 0.4, 0.75)
    data_growth = data_growth_obj.data
    # get fees data for sim
    data_fees_obj = get_filtered_data('fees_normalised', True, 0.65, 0.9)
    data_fees = data_fees_obj.data
    x = Simulate(data_fees, data_growth, valuation_period)
    with Pool(10) as executor:
        start_time = time.perf_counter()
        result = executor.starmap(x.run, ((int(x * (sims / 10)), int(sims / 10)) for x in range(10)))
        finish_time = time.perf_counter()
    print(f"Simulations finished in {finish_time - start_time} seconds")
    return result, data_growth


def compile_sim_results(sim_results: List[Simulate]) -> pd.DataFrame:
    return pd.concat([sim_results[x]._results for x in range(10)]).reset_index(drop=True)


# Run sequentially from 4 to 7
if __name__ == '__main__':
    # Import data from postgre
    summary_data: pd.DataFrame = calculate_growth_percentiles("""select * from d;""", db)
    # Run simulation with multiprocessing pool of 10 processes
    results, d = simulate_with_pool(100000, 4)
    # Compile the simulation results into a single DataFrame
    final = compile_sim_results(results)
    import os

    final.to_pickle(os.path.join(os.getcwd(), 'data4.pickle'))
