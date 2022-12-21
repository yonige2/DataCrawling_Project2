import numpy as np 
import pandas as pd
import os
import random

class Preprocessing:
    def __init__(self):
        self.filePath = 'data'


    def getDataList(self):
        result = []
        df_concat = pd.DataFrame()
        file_names = os.listdir(self.filePath)

        for f in file_names:
            temp_path = f'{self.filePath}/{f}'
            temp = pd.read_csv(temp_path)
            temp = temp[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율', '누적관객수', '순위', 
                '관람객평점', '평론가평점', '최종관객수']]
            df_concat = pd.concat([temp, df_concat])
            result.append(temp)

        df_concat.to_csv('csvSave/concatCSV.csv', index=False, encoding='cp949')
        return result


    def preprocess(self, dataList, concat_csv_path):
        X = []
        y = []
        df_concat = pd.DataFrame()
        concat_csv = pd.read_csv(concat_csv_path, encoding='cp949')
        concat_csv['누적관객수'] = concat_csv['누적관객수'].str.replace(pat=r',', repl='', regex=True)
        concat_csv['누적관객수'] = concat_csv['누적관객수'].apply(pd.to_numeric)

        min_number = concat_csv['누적관객수'].min()
        max_number = concat_csv['누적관객수'].max()
        
        random.shuffle(dataList)

        for data in dataList:
            data['스크린점유율'] = data['스크린점유율'].str.replace(pat=r'%', repl='', regex=True)
            data['상영점유율'] = data['상영점유율'].str.replace(pat=r'%', repl='', regex=True)
            data['좌석점유율'] = data['좌석점유율'].str.replace(pat=r'%', repl='', regex=True)
            data['좌석판매율'] = data['좌석판매율'].str.replace(pat=r'%', repl='', regex=True)
            data['누적관객수'] = data['누적관객수'].str.replace(pat=r',', repl='', regex=True)
            data['최종관객수'] = data['최종관객수'].str.replace(pat=r',', repl='', regex=True)

            data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율', '누적관객수', '순위', '관람객평점', '평론가평점', '최종관객수']] \
                = data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율', '누적관객수', '순위', '관람객평점', '평론가평점', '최종관객수']].apply(pd.to_numeric)

            data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율']] = \
                round(data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율']] / 100, 3)
            
            data[['누적관객수']] = round((data[['누적관객수']] - min_number) / (max_number - min_number) * 5 , 3)
            data[['최종관객수']] = (data[['최종관객수']] - min_number) / (max_number - min_number)

            data[['순위', '관람객평점', '평론가평점']] = round(data[['순위', '관람객평점', '평론가평점']]/20, 3)
           
            df_concat = pd.concat([data, df_concat])
            
            temp_y = data['최종관객수']
            y.append(temp_y[0])
            data = data.drop(['최종관객수'], axis='columns')
            
            temp = data.to_numpy()
            temp = temp.tolist()
            X.append(temp)
        
        df_concat.to_csv('csvSave/data_preprocess.csv', index=False, encoding='cp949')
        


        return X, y


    def preprocess4RunModel(self, test_csv_path, concat_csv_path):
        test_csv = pd.read_csv(test_csv_path)

        concat_csv = pd.read_csv(concat_csv_path, encoding='cp949')
        concat_csv['누적관객수'] = concat_csv['누적관객수'].str.replace(pat=r',', repl='', regex=True)
        concat_csv['누적관객수'] = concat_csv['누적관객수'].apply(pd.to_numeric)

        min_number = concat_csv['누적관객수'].min()
        max_number = concat_csv['누적관객수'].max()

        data = test_csv[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율', '누적관객수', '순위', 
                '관람객평점', '평론가평점']].copy()


        data['스크린점유율'] = data['스크린점유율'].str.replace(pat=r'%', repl='', regex=True)
        data['상영점유율'] = data['상영점유율'].str.replace(pat=r'%', repl='', regex=True)
        data['좌석점유율'] = data['좌석점유율'].str.replace(pat=r'%', repl='', regex=True)
        data['좌석판매율'] = data['좌석판매율'].str.replace(pat=r'%', repl='', regex=True)
        data['누적관객수'] = data['누적관객수'].str.replace(pat=r',', repl='', regex=True)
        
        data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율', '누적관객수', '순위', '관람객평점', '평론가평점']] \
            = data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율', '누적관객수', '순위', '관람객평점', '평론가평점']].apply(pd.to_numeric)
        
        data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율']] = \
            round(data[['스크린점유율', '상영점유율', '좌석점유율', '좌석판매율']] / 100, 3)
         
        data[['누적관객수']] = round(((data[['누적관객수']] - min_number) / (max_number - min_number)) * 5 , 3)

        data[['순위', '관람객평점', '평론가평점']] = round(data[['순위', '관람객평점', '평론가평점']]/20, 3)
        
        temp = data.to_numpy()
        temp = temp.tolist()

        return temp, min_number, max_number