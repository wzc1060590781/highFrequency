import os

import pandas as pd
import numpy as np
import json

def init(index,parameters):
    account = pd.DataFrame(index=index,columns=["total_value","balance","market_value"])
    account.loc[index.to_list()[0],["total_value","balance","market_value"]] = parameters["init_value"],parameters["init_value"],0
    trade_record = pd.DataFrame(columns=["stock_name","amount","open_time","close_time","market_time","open_price","close_price","market_price","direction","status","unrealized_pnl","realized_pnl","unrealized_return","realized_return"])
    return account,trade_record

def trade(parameters):
    threshold = parameters["threshold"]
    firstStock = "600000.XSHG"
    # 新华保险
    secondStock = "601336.XSHG"
    beta = 6.34160734408322
    tax = 0.001
    # merged_origin_5_minute_data = parameters["paths"]["merged_origin_5_minute_data"]
    merged_origin_data_path = parameters["paths"]["merged_origin_5_minute_data"]
    minute_data_df = pd.read_csv(merged_origin_data_path,index_col=0)
    spread_path = r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\spread\600000.XSHG_601336.XSHG_spread.csv"
    spread_df = pd.read_csv(spread_path,index_col=0)
    account,trade_record = init(spread_df.index,parameters)
    spread_df[firstStock]  = minute_data_df[firstStock]
    spread_df[secondStock]  = minute_data_df[secondStock]
    theta = caculateMspreadDivideStd(threshold)
    commition = 0.0002
    for i in range(len(spread_df)):
        row = spread_df.iloc[i]
        date_str_i = spread_df.index.to_list()[i]
        if date_str_i == "2021-10-11 10:35:00":
            print("")
        if i == 0:
            balance = account.iloc[i]["balance"]
            total_value = account.iloc[i]["total_value"]
            market_value = 0
        else:
            balance = account.iloc[i-1]["balance"]
            total_value = account.iloc[i-1]["total_value"]
            market_value = account.iloc[i-1]["market_value"]
        Mspread = row["Mspread"]
        holding_stock_df = trade_record[trade_record["status"] == 1]
        history_stock_df = trade_record[trade_record["status"] == 0]
        # 三种状态1、空仓；2、持有stock_1多头；3、持有stock_2多头
        # 空仓
        if len(holding_stock_df) == 0:
            if Mspread > 0.75 * theta and Mspread < 2*theta:
                first_stock_direction = "long"
                second_stock_direction = "short"
            elif Mspread < -0.75 * theta and Mspread > -2 * theta:
                first_stock_direction = "short"
                second_stock_direction = "long"
            else:
                account.loc[date_str_i,["total_value","market_value","balance"]] = total_value,market_value,balance
                continue
            holding_stock_df,total_value,balance,market_value = open_new_position(first_stock_direction,second_stock_direction,date_str_i,beta,row,firstStock,secondStock,commition,holding_stock_df,balance)
            # trade_record = pd.concat([history_stock_df, holding_stock_df])
        else:
            # 持有stock_2多头
            if len(holding_stock_df[(holding_stock_df["stock_name"]==secondStock)&(holding_stock_df["direction"] == "long")]) == 1:
                # 如果spread > 0 或者 spread < -2* theta则需要平仓
                if Mspread > 0 or Mspread < -2 * theta:
                    holding_stock_df,total_value,balance,market_value = close_holding_position(date_str_i,row,firstStock,secondStock,commition,holding_stock_df,tax,balance)
                else:
                    holding_stock_df, total_value, balance, market_value = update_holding_position(date_str_i,row, firstStock,
                                                                                                   secondStock,
                                                                                                   holding_stock_df,
                                                                                                   commition, balance)
            # 持有stock_1多头
            else:
                if Mspread < 0 or Mspread > 2 * theta:
                    holding_stock_df, total_value, balance, market_value = close_holding_position(date_str_i, row,
                                                                                                  firstStock,
                                                                                                  secondStock,
                                                                                                  commition,
                                                                                                  holding_stock_df, tax,
                                                                                                  balance)

                else:
                    holding_stock_df, total_value, balance, market_value = update_holding_position(date_str_i,row,firstStock,secondStock,holding_stock_df,commition,balance)
        trade_record = pd.concat([history_stock_df, holding_stock_df])
        account.loc[date_str_i, ["total_value","market_value","balance"]] = total_value,market_value,balance
        # account.at[date_str_i,  "market_value"] =  market_value
        # account.at[date_str_i,  "balance"] =  balance
    trade_record.to_csv(r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\trade_record\trade_record.csv")
    account.to_csv(r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_0.9\trade_record\account.csv")

def open_new_position(first_stock_direction,second_stock_direction,date_str_i,beta,row,firstStock,secondStock,commition,holding_df,balance):
    amount = balance // ((beta * 100 * row[firstStock] + 100 * row[secondStock]) * (1 + commition))
    amount_1 = amount * 600
    amount_2 = amount * 100
    market_value = 0
    for stock_name in [firstStock, secondStock]:
        if stock_name == firstStock:
            direction = first_stock_direction
            amount = amount_1
        else:
            direction = second_stock_direction
            amount = amount_2
        open_time = date_str_i
        open_price = row[stock_name]
        trade_record_dict = {
            "stock_name": stock_name,
            "amount": amount,
            "open_time": open_time,
            "close_time": np.nan,
            "market_time": open_time,
            "open_price": open_price,
            "close_price": np.nan,
            "market_price": open_price,
            "direction": direction,
            "status": 1,
            "market_value":open_price*amount,
            "unrealized_pnl": -amount * row[stock_name] * commition,
            "realized_pnl": np.nan,
            "unrealized_return": -commition,
            "realized_return": np.nan,
        }
        balance -= open_price*(1+commition)*amount
        market_value += open_price*amount
        total_value = balance+market_value
        holding_df = holding_df.append(trade_record_dict, ignore_index=True)
    return holding_df,total_value,balance,market_value

def close_holding_position(date_str_i,row,firstStock,secondStock,commition,holding_df,tax,balance):
    for stock_name in [firstStock, secondStock]:
        open_price = holding_df[(holding_df["stock_name"] == stock_name)]["open_price"].iloc[0]
        close_price = row[stock_name]
        close_time = date_str_i
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
        realized_return = realized_pnl/(open_price*(1+commition)*amount)
        market_value = 0
        holding_df.loc[holding_df["stock_name"] == stock_name,["market_time","close_time","close_price","market_price","direction","status",
        "unrealized_pnl","realized_pnl","unrealized_return","realized_return","market_value"]] = [close_time,close_time,close_price,close_price,direction,status,
                          unrealized_pnl,realized_pnl,unrealized_return,realized_return,market_value]
    total_market_value = 0
    total_value = balance

    return holding_df,total_value,balance,total_market_value


def update_holding_position(date_str_i,row,firstStock,secondStock,holding_df,commition,balance):
    total_market_value = 0
    for stock_name in [firstStock, secondStock]:
        open_price = holding_df[(holding_df["stock_name"] == stock_name)]["open_price"].iloc[0]
        close_price = row[stock_name]
        direction = holding_df[(holding_df["stock_name"] == stock_name)]["direction"].iloc[0]
        amount = holding_df[(holding_df["stock_name"] == stock_name)]["amount"].iloc[0]
        if direction == "long":
            unrealized_pnl = (close_price - open_price*(1 + commition)) * amount
            market_value = close_price*amount
            total_market_value += market_value
        else:
            unrealized_pnl = (open_price - close_price - open_price * commition) * amount
            market_value = (2*open_price-close_price) * amount
            total_market_value += market_value
        unrealized_return = unrealized_pnl/((open_price*(1+commition))*amount)
        holding_df.loc[holding_df["stock_name"] == stock_name, ["market_time","market_price","unrealized_pnl", "unrealized_return","market_value"
                                            ]] = [date_str_i,close_price,unrealized_pnl,unrealized_return,market_value]
        total_value = total_market_value + balance
    return holding_df, total_value, balance, total_market_value

def caculateMspreadDivideStd(threshold):
    Mspread_df = pd.read_csv(r"F:\python_project\highFrequency\data\xiezheng\correlation_higher_than_{}\spread\600000.XSHG_601336.XSHG_spread.csv".format(threshold),index_col=0)
    std = caculateStd(Mspread_df["Mspread"].loc["2021-10-08 09:35:00":"2022-04-06 15:00:00"])
    return std

def caculateStd(ser):
    return np.std(ser)


if __name__ == '__main__':
    parameters_path = r"./parameters.json"
    with open(parameters_path,encoding="utf-8") as f:
        parameters = json.load(f)
    trade(parameters)