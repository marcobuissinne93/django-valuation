from django.shortcuts import render
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from . import valuation as valuation
import os
import pandas as pd
import numpy as np
import time
from typing import Dict, Union
import json


class Main(LoginRequiredMixin, View):
    login_url = 'login'

    def get(self, request):
        # results = valuation.run_simulations()
        # print(results.head())
        start = time.perf_counter()

        # val = valuation.Valuation(data, 7, 5, 3_500_000, 0.8, 1_300_000, 250_000)
        # response = get_valuation_data(5, 5, 3_500_000, 0.8, 1_300_000, 250_000, 7, 0.25)
        return render(request, "simulation/simulate.html")


class ValueModel(LoginRequiredMixin, View):
    login_url = 'login'

    def post(self, request):
        resp_json = json.loads(request.body.decode('utf-8'))
        valuation_period = int(resp_json['valuation-period'].replace(',', ''))
        deal_volume = int(resp_json['deal-volume'].replace(',', ''))
        single_investment_size = int(resp_json['invest-size'].replace(',', ''))
        varex = int(resp_json['variable-expense'].replace(',', ''))
        opex = int(resp_json['op-expense'].replace(',', ''))
        discount_rate = float(resp_json['discount-rate'])
        isr = float(resp_json['isr'])
        ebidta = 7
        print(type(valuation_period), type(deal_volume), type(single_investment_size), type(varex), type(opex), type(discount_rate), type(isr))
        response = get_valuation_data(valuation_period, deal_volume, single_investment_size,
                                      isr, opex, varex, ebidta, discount_rate)
        return JsonResponse(response)


def get_valuation_data(valuation_period: int, deal_volume: int,
                       single_investment_value: int, isr: float, opex: int, varex: int,
                       ebidta: int, discount_rate: float) -> Dict[str, Union[int, float]]:
    data = pd.read_pickle(f"./data_store/data{valuation_period}.pickle")
    val = valuation.Valuation(data, valuation_period, deal_volume, single_investment_value, isr, opex, varex)
    valuation_result_set = val.run_valuation('management_fee', np.sum, ebidta, discount_rate)
    total_valuation = valuation_result_set.total_valuation
    net_valuation = valuation_result_set.net_valuation
    cagr = valuation_result_set.cagr
    coc = valuation_result_set.coc
    irr = valuation_result_set.irr
    total_invested = valuation_result_set.total_invested
    fee_valuation = valuation_result_set.fee_valuation
    equity_valuation = valuation_result_set.equity_valuation
    opex = valuation_result_set.opex
    varex = valuation_result_set.varex
    return {'total_valuation': total_valuation,
            'net_valuation': net_valuation,
            'cagr': cagr,
            'coc': coc,
            'irr': irr,
            'total_invested': total_invested,
            'fee_valuation': fee_valuation,
            'equity_valuation': equity_valuation,
            'opex': opex,
            'varex': varex}
