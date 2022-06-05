import pandas as pd
from statsmodels.tsa.arima_model import ARMA
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

def get_ARMA_model(judge_by="aic"):
    # 设置p阶，q阶范围
    # product p,q的所有组合
    # 设置最好的aic为无穷大
    # 对范围内的p,q阶进行模型训练，得到最优模型
    ps = range(0, 6)
    qs = range(0, 6)
    parameters = product(ps, qs)
    parameters_list = list(parameters)
    spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
    best_judy_by = float('inf')
    spread_df = pd.read_csv(spread_path,index_col=0).loc["2021-10-08 09:35:00":"2022-04-06 15:00:00"]
    results = []
    for param in parameters_list:
        # if param[0] == 0 and param[1] == 3:
        try:
            model = ARMA(spread_df['spread'], order=(param[0], param[1])).fit()
        except ValueError:
            print("参数错误：", param)
            continue
        if judge_by == "aic":
            judge_by_score = model.aic
        else:
            judge_by_score = model.bic
        if judge_by_score < best_judy_by:
            best_model = model
            best_judy_by = judge_by_score
            best_param = param
        results.append([param, model.aic])
    results_table = pd.DataFrame(results)

    results_table.columns = ['parameters', judge_by]
    results_table.to_csv(r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\garch\{}.csv".format(judge_by))
    print("最优模型", best_model.summary())
    print(best_param)


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_ACF_PACF():
    spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
    spread_df = pd.read_csv(spread_path,index_col=0)
    plot_acf(spread_df["ARMA_resid_square"],lags=40)
    plot_pacf(spread_df["ARMA_resid_square"],lags=40)
    plt.show()
    # plot_pacf(spread_df["spread"])


def get_ARMA_residual():
    spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
    spread_df = pd.read_csv(spread_path, index_col=0)
    index = spread_df.index
    spread_list = spread_df["spread"].values
    end_loc = np.where(index > '2022-04-06 15:00:00')[0].min()
    history = list(spread_list[:end_loc])
    test = list(spread_list[end_loc:])
    AM = ARMA(history, order=(1, 1))
    model_fit = AM.fit()
    ARMA_resid_list = list(model_fit.resid)
    for t in range(len(test)):
        AM = ARMA(history, order=(1, 1))
        model_fit = AM.fit()
        output = model_fit.forecast()
        yhat = output[0]
        obs = test[t]
        history.append(obs)
        ARMA_resid_list.append(obs - yhat)
        print('predicted=%f, expected=%f' % (yhat, obs))
    spread_df["ARMA_resid"] = ARMA_resid_list
    spread_df["ARMA_resid_square"] = np.square(ARMA_resid_list)
    spread_df.to_csv(spread_path)
    # print(model.resid)



from statsmodels.stats.diagnostic import acorr_ljungbox
def LjungBox_ARMA_residual_square():
    spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
    spread_df = pd.read_csv(spread_path, index_col=0)
    res = acorr_ljungbox(spread_df["ARMA_resid_square"], lags=[i for i in range(1,13)], boxpierce=True)
    cols = ["lbvalue","pvalue","bpvalue","bppvalue"]
    res = pd.DataFrame(res).T
    res.columns = cols
    res.to_csv(r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\garch\ljungbox.csv")

from arch import arch_model
from arch.univariate import GARCH
from itertools import product
l = [1, 2, 3]

list(product(l, l))
def garch(judge_by = "bic"):
    spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
    spread_df = pd.read_csv(spread_path, index_col=0)
    # am = arch_model(spread_df["ARMA_resid_square"],p=2,o=0,q=2)  # 默认模型为GARCH（1，1）
    ps = range(1, 6)
    qs = range(0, 6)
    parameters = product(ps, qs)
    parameters_list = list(parameters)
    best_judy_by = float('inf')
    results = []
    for param in parameters_list:
        am = arch_model(spread_df["ARMA_resid"],vol="GARCH",p = param[0],o=0,q=param[1])  # 默认模型为GARCH（1，2）
        model = am.fit(update_freq=0)  # 估计参数
        if judge_by == "aic":
            judge_by_score = model.aic
        else:
            judge_by_score = model.bic
        if judge_by_score < best_judy_by:
            best_model = model
            best_judy_by = judge_by_score
            best_param = param
        results.append([param, model.aic])
    results_table = pd.DataFrame(results)
    results_table.columns = ['parameters', judge_by]
    results_table.to_csv(r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\garch\garch_{}.csv".format(judge_by))
    print("最优模型", best_model.summary())
    print(best_model)

def model_forcast():
    spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
    spread_df = pd.read_csv(spread_path, index_col=0)
    # spread_df.index = pd.DatetimeIndex(spread_df.index.to_list())
    am = arch_model(spread_df["ARMA_resid"],p=1,q=2)  # 默认模型为GARCH（1，1）
    index = spread_df.index
    end_loc = np.where(index > '2022-04-06 15:00:00')[0].min()
    for i in range(912):
        # first_obs = datetime.strptime("2021-10-08 09:35:00", "%Y-%m-%d %H:%M:%S")
        # last_obs = datetime.strptime("2022-04-06 15:00:00", "%Y-%m-%d %H:%M:%S")
        am_model  = am.fit(first_obs=i,last_obs=i+end_loc,update_freq=0)
        # am  = am.fit(update_freq=0)
        var = am_model.forecast().variance.iloc[i+end_loc-1]['h.1']
        theta = (spread_df["spread"]-spread_df.loc[:"2022-04-06 15:00:00"]["spread"].mean()).iloc[i+end_loc]/np.sqrt(var)
        print(theta)
        # print(am.fit(update_freq=0))

# def arma_forcast():
#     spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
#     spread_df = pd.read_csv(spread_path, index_col=0)
#     model = ARMA(spread_df['spread'], order=(1, 1)).fit()
#     index = spread_df.index
#     end_loc = np.where(index >= '2022-04-06 09:35:00')[0].min()
#     for i in range(48):
#         # first_obs = datetime.strptime("2021-10-08 09:35:00", "%Y-%m-%d %H:%M:%S")
#         # last_obs = datetime.strptime("2022-04-06 15:00:00", "%Y-%m-%d %H:%M:%S")
#         am_model = am.fit(first_obs=i, last_obs=i + end_loc, update_freq=0)
#         # am  = am.fit(update_freq=0)
#         var = am_model.forecast().variance.iloc[i + end_loc - 1]['h.1']
#         theta = (spread_df["spread"] - spread_df["spread"].mean()).iloc[i + end_loc] / np.sqrt(var)
#         print(theta)
#         # print(am.fit(update_freq=0))
if __name__ == '__main__':
    # get_ARMA_model("bic")
    model_forcast()
    # get_ARMA_residual()