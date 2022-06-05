import json
import os
import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
# mpl.rcParams["font.san-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False




class CointeGration():
    def __init__(self,parameters,threshold):
        self.parameters = parameters
        self.threshold = threshold
        self.CointeGration_paramters = parameters["paths"]["CointeGration"]
        self.strategy_dir = parameters["paths"]["CointeGration"]["strategy_dir"].format(threshold)



    def caculateCorrelation(self):
        merged_origin_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        correlation_relative_path = self.CointeGration_paramters["correlation_path"]
        correlation_path = os.path.join(self.strategy_dir,correlation_relative_path)
        df = pd.read_csv(merged_origin_data_path,index_col=0)
        df.corr().to_csv(correlation_path)


    def getCorrelationGreaterThan(self):
        correlation_relative_path = self.CointeGration_paramters["correlation_path"]
        correlation_path = os.path.join(self.strategy_dir, correlation_relative_path)
        chosed_stock_relative_path = self.CointeGration_paramters["chosed_stock_relative_path"]
        chosed_stock_path = os.path.join(self.strategy_dir,chosed_stock_relative_path)
        df = pd.read_csv(correlation_path,index_col=0)
        res_list = []
        for index,row in df.iterrows():
            for col in row.index:
                if row.loc[col] >= self.threshold and row.loc[col] != 1:
                    res_list.append([index,col])
        new_res_list = []
        for i in res_list:
            if i in new_res_list or [i[1],i[0]] in new_res_list:
                continue
            else:
                new_res_list.append(i)
        pd.DataFrame(new_res_list,columns=["firstStockCode","secondStockCode"]).to_csv(chosed_stock_path)

    def chosedStockADF(self,N=0):
        chosed_stock_relative_path = self.CointeGration_paramters["chosed_stock_relative_path"]
        chosed_stock_path = os.path.join(self.strategy_dir, chosed_stock_relative_path)
        ADF_results_relative_dir = self.CointeGration_paramters["ADF_results_relative_dir"]
        ADF_results_dir = os.path.join(self.strategy_dir,ADF_results_relative_dir)

        if not os.path.exists(ADF_results_dir):
            os.mkdir(ADF_results_dir)
        ADF_results_path = os.path.join(ADF_results_dir,"ADF_results_{}.csv".format(N))
        df = pd.read_csv(chosed_stock_path,index_col=0)
        merged_origin_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        mergeOrigin5minuteDf = pd.read_csv(merged_origin_data_path,index_col=0)
        firstStockSer = df["firstStockCode"].unique()
        secondStockSer = df["secondStockCode"].unique()
        stock_list = list(set(list(firstStockSer) + list(secondStockSer)))
        res_list = []
        for stock in stock_list:
            priceSer = mergeOrigin5minuteDf[stock]
            priceSer.bfill(inplace=True)
            for i in range(N):
                priceSer = priceSer.diff(1)
            priceSer.dropna(inplace=True)
            res =ts.adfuller(priceSer)
            res_list.append(list(res[:4])+[stock])
        pd.DataFrame(res_list,columns=["T","P","Lags Used","NumberObservationsUsed","stockName"]).to_csv(ADF_results_path)


    def caculateResidual(self):
        chosed_stock_relative_path = self.CointeGration_paramters["chosed_stock_relative_path"]
        chosed_stock_path = os.path.join(self.strategy_dir, chosed_stock_relative_path)
        chosed_stock_df = pd.read_csv(chosed_stock_path,index_col=0)
        generate_beta_alpha_index = chosed_stock_df.apply(lambda x:x["firstStockCode"]+"_"+x["secondStockCode"],axis=1)
        beta_alpha_df = pd.DataFrame(index=generate_beta_alpha_index,columns=["alpha","beta"])
        merged_origin_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        residual_relative_dir = self.CointeGration_paramters["residual_relative_dir"]
        residual_dir = os.path.join(self.strategy_dir, residual_relative_dir)
        if not os.path.exists(residual_dir):
            os.mkdir(residual_dir)
        minute_data_df = pd.read_csv(merged_origin_data_path,index_col=0)
        residual_df = pd.DataFrame(index=minute_data_df.index)
        chosed_stock_df.apply(self.coreCaculateResidual, args = (minute_data_df, residual_df,beta_alpha_df), axis=1)
        residual_df.to_csv(os.path.join(residual_dir,"residual_df.csv"))
        beta_alpha_df.to_csv(os.path.join(residual_dir,"beta_alpha_df.csv"))



    def coreCaculateResidual(self,ser, minute_data_df, residual_df,beta_alpha_df):
        first_stock = ser["firstStockCode"]
        second_stock = ser["secondStockCode"]
        first_stock_ser = minute_data_df[first_stock]
        second_stock_ser = minute_data_df[second_stock]
        x = sm.add_constant(first_stock_ser)
        y = second_stock_ser
        model = sm.OLS(y,x)
        results = model.fit()
        beta_alpha_df.loc[first_stock+"_"+second_stock] = [results.params.loc["const"],results.params.loc[first_stock]]
        print(results.params)
        residual = second_stock_ser-first_stock_ser*results.params.loc[first_stock]-results.params.loc["const"]
        residual_df[first_stock+"_"+second_stock] = residual





    def caculateCoint(self):
        chosed_stock_relative_path = self.CointeGration_paramters["chosed_stock_relative_path"]
        chosed_stock_path = os.path.join(self.strategy_dir, chosed_stock_relative_path)
        df = pd.read_csv(chosed_stock_path, index_col=0)
        merged_origin_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        mergeOrigin5minuteDf = pd.read_csv(merged_origin_data_path, index_col=0)

        CointRes_relative_path = self.CointeGration_paramters["CointRes_relative_path"]
        CointResPath = os.path.join(self.strategy_dir, CointRes_relative_path)
        res_list = []
        for index,row in df.iterrows():
            firstStockCode = row["firstStockCode"]
            secondStockCode = row["secondStockCode"]
            res = ts.coint(mergeOrigin5minuteDf[firstStockCode],mergeOrigin5minuteDf[secondStockCode])
            res_list.append([firstStockCode+"_"+secondStockCode,res[0],res[1]])
        pd.DataFrame(res_list, columns=["stockName", "T", "P"]).to_csv(
            CointResPath)

    def checkResidualADF(self):
        residual_relative_dir = self.CointeGration_paramters["residual_relative_dir"]
        residual_dir = os.path.join(self.strategy_dir, residual_relative_dir)
        residual_path = os.path.join(residual_dir,"residual_df.csv")
        residual_df = pd.read_csv(residual_path,index_col=0)
        residual_ADF_results_path = os.path.join(residual_dir,"residual_ADF_results.csv")
        res_list = []
        residual_df.apply(self.coreCheckResidualADF,args = (res_list,))
        pd.DataFrame(res_list, columns=["T", "P", "Lags Used", "NumberObservationsUsed", "stockName"]).to_csv(
            residual_ADF_results_path)


    def coreCheckResidualADF(self,ser,res_list):
        res =ts.adfuller(ser,maxlag=0)
        res_list.append(list(res[:4])+[ser.name])

    def caculateSpread(self):
        # 浦发银行
        firstStock = "600000.XSHG"
        # 新华保险
        secondStock = "601336.XSHG"
        residual_relative_dir = self.CointeGration_paramters["residual_relative_dir"]
        residual_dir = os.path.join(self.strategy_dir, residual_relative_dir)
        beta_alpha_path = os.path.join(residual_dir, "beta_alpha_df.csv")
        alpha_beta_df = pd.read_csv(beta_alpha_path,index_col=0)
        merged_origin_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        minute_data_df = pd.read_csv(merged_origin_data_path, index_col=0).loc["2021-10-08 09:35:00":]
        beta = alpha_beta_df.loc[firstStock+"_"+secondStock]["beta"]
        spread = minute_data_df[secondStock] - beta* minute_data_df[firstStock]
        Mspread = spread - np.mean(spread.loc["2021-10-08 09:35:00":"2022-04-06 15:00:00"])
        Mspread_df = pd.DataFrame(index=Mspread.index)
        Mspread_df["spread"] = spread
        Mspread_df["Mspread"] = Mspread
        spread_relative_dir = self.CointeGration_paramters["spread_relative_dir"]
        spread_dir  = os.path.join(self.strategy_dir,spread_relative_dir)
        Mspread_df.to_csv(os.path.join(spread_dir,"{}_{}_spread.csv".format(firstStock,secondStock)))


    def plot_name(self):
        firstStock = "600000.XSHG"
        # 新华保险
        secondStock = "600050.XSHG"
        merged_origin_data_path = self.parameters["paths"]["merged_origin_5_minute_data"]
        minute_data_df = pd.read_csv(merged_origin_data_path, index_col=0)
        p1, = plt.plot(minute_data_df[firstStock])
        p2, = plt.plot(minute_data_df[secondStock])
        plt.xticks(minute_data_df.index.to_list()[::480],rotation=45)
        l1 = plt.legend([p1,p2],["万华化学","中国人寿"])
        plt.show()

def plot(threshold):
    firstStock = "600000.XSHG"
    # 新华保险
    secondStock = "601336.XSHG"
    Mspread_df = pd.read_csv(r"E:\高频交易\data\xiezheng\correlation_higher_than_{}\spread\{}_{}_spread.csv".format(threshold,firstStock,secondStock),index_col=0)
    plt.plot(list(range(len(Mspread_df))),Mspread_df["Mspread"]/np.std(Mspread_df["Mspread"]))
    plt.show()

def caculate_pecentile(threshold=0.9):
    firstStock = "600000.XSHG"
    # 新华保险
    secondStock = "601336.XSHG"
    Mspread_df = pd.read_csv(r"E:\高频交易\data\xiezheng\correlation_higher_than_{}\spread\{}_{}_spread.csv".format(threshold,firstStock,secondStock),index_col=0)
    print(np.mean(Mspread_df["Mspread"]))
    print(np.percentile(Mspread_df["Mspread"],25))
    print(np.percentile(Mspread_df["Mspread"],75))
    print(np.median(Mspread_df["Mspread"]))
    print(np.var(Mspread_df["Mspread"]))
    print(np.std(Mspread_df["Mspread"]))



if __name__ == '__main__':
    parameters_path = r"./parameters.json"
    with open(parameters_path,encoding="utf-8") as f:
        parameters = json.load(f)
    # parameters = json.loads(parameters_path)
    # getCorrelationGreaterThan()
    # chosedStockADF(1)
    # chosedStockADF(N=1,threshold=0.85)
    # caculateResidual(0.9)
    # caculateSpread(0.9)
    # plot_name()
    # plot(threshold=0.9)
    cg = CointeGration(parameters,0.9)
    cg.caculateSpread()