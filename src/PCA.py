# // 用python实现主成分分析（PCA）
import json
import os

import numpy as np
import pandas as pd
# from numpy.linalg import eig
# from sklearn import preprocessing
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import  statsmodels.api as sm

def pca(X,k):
    X = X - X.mean(axis = 0) #向量X去中心化
    X_cov = np.cov(X.T, ddof = 0) #计算向量X的协方差矩阵，自由度可以选择0或1
    eigenvalues,eigenvectors = eig(X_cov) #计算协方差矩阵的特征值和特征向量
    klarge_index = eigenvalues.argsort()[-k:][::-1] #选取最大的K个特征值及其特征向量
    k_eigenvectors = eigenvectors[klarge_index] #用X与特征向量相乘
    return np.dot(X, k_eigenvectors.T)

# // An highlighted block
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import eig
#matplotlib inline

# iris = load_iris()
# X = iris.data
class PCAStrategy():
    def __init__(self,parameters):
        self.parameters = parameters
        self.PCA_paramters = parameters["paths"]["PCA"]
        self.strategy_dir = parameters["paths"]["PCA"]["strategy_dir"]

    def percentage2n(self,eigVals,percentage):
        sortArray = np.sort(eigVals)
        sortArray = sortArray[-1::-1]
        arraySum = sum(sortArray)
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum*percentage:
                return num

    def caculate_PCA(self,X):
        X = (X - X.mean())/X.std()
        X_correlation = X.corr()
        X_correlation = X_correlation.values
        eigenvalues,eigenvectors = eig(X_correlation)
        PAC_relative_results_dir = self.PCA_paramters["PAC_relative_results"]
        PAC_results_dir = os.path.join(self.strategy_dir,PAC_relative_results_dir)
        eigenvectorn_path = os.path.join(PAC_results_dir,"eigenvectorn.csv")
        explained_variance_path = os.path.join(PAC_results_dir, "explained_variance.csv")
        pd.DataFrame(eigenvalues).to_csv(explained_variance_path)
        explained_variance_ratio_path = os.path.join(PAC_results_dir, "explained_variance_ratio_.csv")
        pd.DataFrame(eigenvalues/np.sum(eigenvalues)).to_csv(explained_variance_ratio_path)
        pd.DataFrame(eigenvectors).to_csv(eigenvectorn_path)
        plt.plot(eigenvalues)
        plt.show()


    def get_5_minute_data(self):
        start_date = self.PCA_paramters["start_date"]
        end_date = self.PCA_paramters["end_date"]
        minute_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        df = pd.read_csv(minute_data_path,index_col=0).loc[start_date+" 09:35:00":end_date+" 15:00:00"]
        # df = self.delete_stock(df)
        return df

    def delete_stock(self,df):
        delete_stock_list = self.PCA_paramters["delete_stocks"]
        return df.drop(delete_stock_list,axis=1)

    def build_OLS_model(self):
        start_date = self.PCA_paramters["start_date"]
        end_date = self.PCA_paramters["end_date"]
        stock = ["601066.XSHG", "600585.XSHG", "600745.XSHG"]
        A50_index_path = self.parameters["paths"]["index_path"]
        OLS_relative_results_dir = self.PCA_paramters["OLS_relative_results"]
        OLS_results_dir = os.path.join(self.strategy_dir, OLS_relative_results_dir)
        minute_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        A50_df = pd.read_csv(A50_index_path, index_col=0).loc[start_date+" 09:35:00":end_date+" 15:00:00"]
        stock_data = pd.read_csv(minute_data_path, index_col=0)
        df = pd.DataFrame(index=A50_df.index)
        df["A50"] = A50_df["close"]
        df[stock] = stock_data[stock]
        x = df[stock]
        x = sm.add_constant(x)
        y = df["A50"]
        model = sm.OLS(y, x)
        res = model.fit()
        res.params.to_csv(os.path.join(OLS_results_dir,"beta.csv"))

    def caculateSpread(self):
        # 浦发银行
        firstStock = "601066.XSHG"
        # 新华保险
        secondStock = "600585.XSHG"
        third_Stock = "600745.XSHG"
        start_date = self.PCA_paramters["start_date"]
        end_date = self.PCA_paramters["end_date"]
        A50_index_path = self.parameters["paths"]["index_path"]
        A50_ser = pd.read_csv(A50_index_path, index_col=0)["close"].loc[start_date+" 09:35:00":end_date+" 15:00:00"]
        OLS_relative_results_dir = self.PCA_paramters["OLS_relative_results"]
        OLS_results_dir = os.path.join(self.strategy_dir, OLS_relative_results_dir)
        beta_alpha_path = os.path.join(OLS_results_dir, "beta.csv")
        alpha_beta_df = pd.read_csv(beta_alpha_path, index_col=0)
        merged_origin_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        minute_data_df = pd.read_csv(merged_origin_data_path, index_col=0).loc[start_date+" 09:35:00":end_date+" 15:00:00"]
        spread = A50_ser - alpha_beta_df.loc[firstStock,"0"]*minute_data_df[firstStock] - \
                 alpha_beta_df.loc[secondStock,"0"]*minute_data_df[secondStock] - alpha_beta_df.loc[third_Stock,"0"]*minute_data_df[third_Stock]
        Mspread = spread - np.mean(spread)
        Mspread_df = pd.DataFrame(index=Mspread.index)
        Mspread_df["spread"] = spread
        Mspread_df["Mspread"] = Mspread
        spread_relative_dir = self.PCA_paramters["spread_relative_dir"]
        spread_dir = os.path.join(self.strategy_dir, spread_relative_dir)
        Mspread_df.to_csv(os.path.join(spread_dir, "{}_{}_{}_spread.csv".format(firstStock, secondStock,third_Stock)))

    def main(self):
        minute_data = self.get_5_minute_data()
        self.caculate_PCA(minute_data)

    def init(self,index):
        account = pd.DataFrame(index=index, columns=["total_value", "balance", "market_value"])
        account.loc[index.to_list()[0], ["total_value", "balance", "market_value"]] = self.parameters["init_value"], \
                                                                                      self.parameters["init_value"], 0
        trade_record = pd.DataFrame(
            columns=["stock_name", "amount", "open_time", "close_time", "market_time", "open_price", "close_price",
                     "market_price", "direction", "status", "unrealized_pnl", "realized_pnl", "unrealized_return",
                     "realized_return"])
        return account, trade_record

    def caculateMspreadDivideStd(self):
        spread_relative_dir = self.PCA_paramters["spread_relative_dir"]
        spread_dir = os.path.join(self.strategy_dir, spread_relative_dir)
        firstStock = "601066.XSHG"
        # 新华保险
        secondStock = "600585.XSHG"
        third_Stock = "600745.XSHG"
        Mspread_df = pd.read_csv(os.path.join(spread_dir, "{}_{}_{}_spread.csv".format(firstStock, secondStock,third_Stock)),
                                 index_col=0)
        std = self.caculateStd(Mspread_df["Mspread"])
        return std

    def caculateStd(self,ser):
        return np.std(ser)


    def trade(self):
        start_date = self.PCA_paramters["start_date"]
        end_date = self.PCA_paramters["end_date"]
        index = "A50"
        firstStock = "601066.XSHG"
        # 新华保险
        secondStock = "600585.XSHG"
        third_Stock = "600745.XSHG"
        beta_dict = {
                        index:1,
                        firstStock:33,
                        secondStock:24,
                        third_Stock:7
                     }
        position_status = 0
        tax = 0.001
        merged_origin_5_minute_data_path = parameters["paths"]["merged_origin_5_minute_data"]
        minute_data_df = pd.read_csv(merged_origin_5_minute_data_path, index_col=0).loc[start_date + " 09:35:00":end_date + " 15:00:00"]
        account, trade_record = self.init(minute_data_df.index)
        spread_relative_dir = self.PCA_paramters["spread_relative_dir"]
        spread_dir = os.path.join(self.strategy_dir, spread_relative_dir)
        spread_df = pd.read_csv(
            os.path.join(spread_dir, "{}_{}_{}_spread.csv".format(firstStock, secondStock, third_Stock)),
            index_col=0)
        A50_index_path = self.parameters["paths"]["index_path"]
        start_date = self.PCA_paramters["start_date"]
        end_date = self.PCA_paramters["end_date"]
        A50_ser = pd.read_csv(A50_index_path, index_col=0)["close"].loc[start_date + " 09:35:00":end_date + " 15:00:00"]
        spread_df[firstStock] = minute_data_df[firstStock]
        spread_df[secondStock] = minute_data_df[secondStock]
        spread_df[third_Stock] = minute_data_df[third_Stock]
        spread_df["A50"] = A50_ser
        theta = self.caculateMspreadDivideStd()
        commition = 0.0001
        for i in range(len(spread_df)):
            row = spread_df.iloc[i]
            date_str_i = spread_df.index.to_list()[i]
            if i == 0:
                balance = account.iloc[i]["balance"]
                total_value = account.iloc[i]["total_value"]
                market_value = 0
            else:
                balance = account.iloc[i - 1]["balance"]
                total_value = account.iloc[i - 1]["total_value"]
                market_value = account.iloc[i - 1]["market_value"]
            Mspread = row["Mspread"]
            holding_stock_df = trade_record[trade_record["status"] == 1]
            history_stock_df = trade_record[trade_record["status"] == 0]
            # 三种状态1、空仓；2、持有stock_1多头；3、持有stock_2多头
            # 空仓
            if len(holding_stock_df) == 0:
                # 指数被高估，应该卖出指数，卖出"600585.XSHG"，卖出600745.XSHG
                if Mspread > 0.75 * theta and Mspread < 2 * theta:
                    direction_dic = {
                        index:"short",
                        firstStock:"long",
                        secondStock:"short",
                        third_Stock:"short"
                    }
                elif Mspread < -0.75 * theta and Mspread > -2 * theta:
                    direction_dic = {
                        index: "long",
                        firstStock: "short",
                        secondStock: "long",
                        third_Stock: "long"
                    }

                else:
                    account.loc[
                        date_str_i, ["total_value", "market_value", "balance"]] = total_value, market_value, balance
                    continue
                holding_stock_df, total_value, balance, market_value = self.open_new_position(direction_dic, beta_dict, date_str_i, row,
                          commition, holding_stock_df, balance)
            else:
                # 持有stock_2多头
                if len(holding_stock_df[(holding_stock_df["stock_name"] == secondStock) & (
                        holding_stock_df["direction"] == "long")]) == 1:
                    if Mspread > 0 or Mspread < -2 * theta:
                        holding_stock_df, total_value, balance, market_value = self.close_holding_position(date_str_i, row,
                                                                                                      beta_dict,
                                                                                                      commition,
                                                                                                      holding_stock_df,
                                                                                                      tax, balance)
                    else:
                        holding_stock_df, total_value, balance, market_value = self.update_holding_position(date_str_i, row, beta_dict, holding_stock_df, commition, balance)
                # 持有stock_1多头
                else:
                    if Mspread < 0 or Mspread > 2 * theta:
                        holding_stock_df, total_value, balance, market_value = self.close_holding_position(date_str_i, row, beta_dict, commition, holding_stock_df, tax, balance)

                    else:
                        holding_stock_df, total_value, balance, market_value = self.update_holding_position(date_str_i, row, beta_dict, holding_stock_df, commition, balance)
            trade_record = pd.concat([history_stock_df, holding_stock_df])
            account.loc[date_str_i, ["total_value", "market_value", "balance"]] = total_value, market_value, balance
        trade_record_relative_dir = self.PCA_paramters["trade_record_relative_dir"]
        trade_record_dir = os.path.join(self.strategy_dir,trade_record_relative_dir)
        trade_record.to_csv(os.path.join(trade_record_dir,"trade_record.csv"))
        account.to_csv(os.path.join(trade_record_dir,"account.csv"))

    def open_new_position(self,direction_dic, beta_dict, date_str_i, row,
                          commition, holding_df, balance):
        temp_sum = 0
        for key,value in direction_dic.items():
            temp_sum += beta_dict[key]*row[key]

        amount = balance // temp_sum
        market_value = 0
        for stock_name,value in beta_dict.items():
            direction = direction_dic[stock_name]
            amount_temp = amount*abs(value)
            open_time = date_str_i
            open_price = row[stock_name]
            trade_record_dict = {
                "stock_name": stock_name,
                "amount": amount_temp,
                "open_time": open_time,
                "close_time": np.nan,
                "market_time": open_time,
                "open_price": open_price,
                "close_price": np.nan,
                "market_price": open_price,
                "direction": direction,
                "status": 1,
                "market_value": open_price * amount_temp,
                "unrealized_pnl": -amount_temp * row[stock_name] * commition,
                "realized_pnl": np.nan,
                "unrealized_return": -commition,
                "realized_return": np.nan,
            }
            balance -= open_price * (1 + commition) * amount_temp
            market_value += open_price * amount_temp
            holding_df = holding_df.append(trade_record_dict, ignore_index=True)
        total_value = balance + market_value
        return holding_df, total_value, balance, market_value

    def close_holding_position(self,date_str_i, row, beta_dict, commition, holding_df, tax, balance):
        # for stock_name in holding_df.iterrows():
        for stock_name in beta_dict.keys():
            # stock_name = holding_row["stock_name"]
            open_price = holding_df[(holding_df["stock_name"] == stock_name)]["open_price"].iloc[0]
            close_price = row[stock_name]
            close_time = date_str_i
            # direction = value
            direction = holding_df[(holding_df["stock_name"] == stock_name)]["direction"].iloc[0]
            amount = holding_df[(holding_df["stock_name"] == stock_name)]["amount"].iloc[0]
            status = 0
            if direction == "long":
                realized_pnl = (row[stock_name] * (1 - commition - tax) - open_price * (1 + commition)) * amount
                balance += close_price * (1 - commition - tax) * amount
            else:
                realized_pnl = (open_price - close_price - open_price * commition - close_price * (
                        commition + tax)) * amount
                balance += (2 * open_price - close_price - close_price * (commition + tax)) * amount
            unrealized_pnl = 0
            unrealized_return = 0
            realized_return = realized_pnl / (open_price * (1 + commition) * amount)
            market_value = 0
            holding_df.loc[
                holding_df["stock_name"] == stock_name, ["market_time", "close_time", "close_price", "market_price",
                                                         "direction", "status",
                                                         "unrealized_pnl", "realized_pnl", "unrealized_return",
                                                         "realized_return", "market_value"]] = [close_time, close_time,
                                                                                                close_price,
                                                                                                close_price, direction,
                                                                                                status,
                                                                                                unrealized_pnl,
                                                                                                realized_pnl,
                                                                                                unrealized_return,
                                                                                                realized_return,
                                                                                                market_value]
        total_market_value = 0
        total_value = balance
        return holding_df, total_value, balance, total_market_value

    def update_holding_position(self,date_str_i, row, beta_dict, holding_df, commition, balance):
        total_market_value = 0
        for stock_name in beta_dict.keys():
            open_price = holding_df[(holding_df["stock_name"] == stock_name)]["open_price"].iloc[0]
            close_price = row[stock_name]
            direction = holding_df[(holding_df["stock_name"] == stock_name)]["direction"].iloc[0]
            amount = holding_df[(holding_df["stock_name"] == stock_name)]["amount"].iloc[0]
            if direction == "long":
                unrealized_pnl = (close_price - open_price * (1 + commition)) * amount
                market_value = close_price * amount
                total_market_value += market_value
            else:
                unrealized_pnl = (open_price - close_price - open_price * commition) * amount
                market_value = (2 * open_price - close_price) * amount
                total_market_value += market_value
            unrealized_return = unrealized_pnl / ((open_price * (1 + commition)) * amount)
            holding_df.loc[holding_df["stock_name"] == stock_name, ["market_time", "market_price", "unrealized_pnl",
                                                                    "unrealized_return", "market_value"
                                                                    ]] = [date_str_i, close_price, unrealized_pnl,
                                                                          unrealized_return, market_value]
        total_value = total_market_value + balance
        return holding_df, total_value, balance, total_market_value





def build_ECM():
    beta_path = r"E:\高频交易\data\PCA\OLS\beta.csv"
    beta_df = pd.read_csv(beta_path,index_col=0)
    A50_index_path = r"E:\高频交易\data\A50.csv"
    stock_data_path = r"E:\高频交易\data\merged_origin_5_minute_data\merged_origin_5_minute_data.csv"
    A50_df = pd.read_csv(A50_index_path, index_col=0)
    stock_data = pd.read_csv(stock_data_path, index_col=0)
    df = pd.DataFrame(index=A50_df.index)
    df["A50"] = A50_df["close"]
    for stock in beta_df.index.to_list()[1:]:
        df[stock] = stock_data[stock]
        df["delta_" + stock] = df[stock].diff(1)
    df["delta_A50"] = df["A50"].diff(1)
    res = None
    for i in range(1,len(beta_df.index.to_list())):
        stock = beta_df.index.to_list()[i]
        if res is None:
            res = beta_df.loc[stock][0]*df[stock]
        else:
            res += beta_df.loc[stock][0]*df[stock]
    df["ECM"] = df["A50"] - res - beta_df.loc["const"][0]
    df["ECM"] = df["ECM"].shift(1)
    stock_list = ["600585.XSHG", "601066.XSHG", "601899.XSHG"]
    col_list = ["delta_" + stock for stock in stock_list]
    col_list.append("ECM")
    x = df[col_list]
    x.dropna(axis=0,inplace=True)
    x = sm.add_constant(x)
    y = df["delta_A50"]
    y.dropna(axis=0, inplace=True)
    model = sm.OLS(y, x)
    res = model.fit()
    res.params.to_csv(r"E:\高频交易\data\PCA\OLS\ECM.csv")


# def main():
#     minute_data = get_5_minute_data()
#     caculate_PCA(minute_data)
    # k = 6
    # caculate_PCA(minute_data)


if __name__ == '__main__':
    parameters_path = r"./parameters.json"
    with open(parameters_path, encoding="utf-8") as f:
        parameters = json.load(f)
    PCAS = PCAStrategy(parameters)
    PCAS.trade()
    # build_ECM()
    # build_ECM_model()
    # caculate_PCA()
    # df = pd.read_csv(r"E:\高频交易\data\PCA\PAC_results\loadings.csv",index_col=0)
    # (df[df.columns[0]]*df[df.columns[0]]).sum()