import json
import os
import pandas as pd

def mergeOrigin5minuteData(parameters):
    origin_5_minute_data = parameters["paths"]["origin_5_minute_data"]
    merged_origin_5_minute_df = None
    end_date = parameters["paths"]["end_date_of_data"]
    destination_path = parameters["paths"]["merged_origin_5_minute_data"]
    for stock_dir in os.listdir(origin_5_minute_data):
        stock_dir_path = os.path.join(origin_5_minute_data,stock_dir)
        data_file_path = os.path.join(stock_dir_path,end_date+".csv")
        stock_df = pd.read_csv(data_file_path,index_col=0)
        if merged_origin_5_minute_df is None:
            merged_origin_5_minute_df = pd.DataFrame(index= stock_df.index,columns=[os.listdir(origin_5_minute_data)])
            merged_origin_5_minute_df[[stock_dir]] = stock_df["close"]
        else:
            merged_origin_5_minute_df[[stock_dir]] = stock_df["close"]
    merged_origin_5_minute_df.to_csv(destination_path)



def mergeOrigin5minuteData2():
    origin_5_minute_data = r"C:\Users\Administrator\Desktop\A50_5minute_data"
    merged_origin_5_minute_df = None
    end_date = "2022-05-27"
    # destination_path = parameters["paths"]["merged_origin_5_minute_data"]
    for stock_dir in os.listdir(origin_5_minute_data):
        stock_dir_path = os.path.join(origin_5_minute_data,stock_dir)
        data_file_path = os.path.join(stock_dir_path,end_date+".csv")
        stock_df = pd.read_csv(data_file_path,index_col=0)
        if merged_origin_5_minute_df is None:
            merged_origin_5_minute_df = pd.DataFrame(index= stock_df.index,columns=[os.listdir(origin_5_minute_data)])
            merged_origin_5_minute_df[[stock_dir]] = stock_df["close"]
        else:
            merged_origin_5_minute_df[[stock_dir]] = stock_df["close"]
    print(merged_origin_5_minute_df)
    # merged_origin_5_minute_df.to_csv(destination_path)




def delete_stock(stock,parameters):
    merged_origin_5_minute_data_path = parameters["paths"]["merged_origin_5_minute_data"]
    merged_origin_5_minute_data = pd.read_csv(merged_origin_5_minute_data_path,index_col=0)
    stock_ser = merged_origin_5_minute_data[stock]
    stock_ser.dropna(inplace=True)
    index_path = parameters["paths"]["index_path"]
    index_df = pd.read_csv(index_path,index_col=0)
    if len(index_df) != len(stock_ser):
        print(stock)


if __name__ == '__main__':
    path = "F:\python_project\highFrequency\data\merged_origin_5_minute_data\merged_origin_5_minute_data.csv"
    path2 = r"C:\Users\Administrator\Desktop\merged_data.csv"
    df = pd.read_csv(path,index_col=0)
    df2 = pd.read_csv(path2,index_col=0)
    df = pd.concat([df,df2])
    df = df[~df.index.duplicated(keep='first')]
    print(df.to_csv(path))

    # mergeOrigin5minuteData2()
    # parameters_path = r"./parameters.json"
    # with open(parameters_path, encoding="utf-8") as f:
    #     parameters = json.load(f)
    # # mergeOrigin5minuteData(parameters)
    # dir = "E:\????????????\data\A50_5minute_data"
    # for stock in os.listdir(dir):
    #     stock_dir = os.path.join(dir,stock)
    #     stock_path = os.path.join(stock_dir,"2022-04-06.csv")
    #     stock_df = pd.read_csv(stock_path,index_col=0)
    #     res = stock_df.dropna(how="all",axis=0)
    #     if len(res) != len(stock_df):
    #         print(stock)
        # delete_stock(stock,parameters)