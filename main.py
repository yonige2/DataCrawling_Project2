from preprocessing import Preprocessing
from model import PredictModel
from runModel import RunModel


preprocessing = Preprocessing()
predictModel = PredictModel()
runModel = RunModel()
movie = '닥터 스트레인지 대혼돈의 멀티버스' # '블랙 팬서 와칸다 포에버' '스파이더맨 노 웨이 홈'
test_csv_path = f'testData\{movie}.csv'
concat_csv_path = 'csvSave/concatCSV.csv'
modelPath = 'model/bestModel.h5'

def model_Train():
    dataList = preprocessing.getDataList()
    X, y = preprocessing.preprocess(dataList, concat_csv_path)
    
    row, column, X_train, y_train, X_test, y_test = predictModel.loadData(X,y)
    predictModel.model(row, column, X_train, y_train, X_test, y_test)


def model_Run():
    data, minNumber, maxNumber = preprocessing.preprocess4RunModel(test_csv_path, concat_csv_path)
    model = runModel.loadModel(modelPath)
    X = runModel.preprocessData(data)
    result = runModel.runModel(model, X, minNumber, maxNumber)
    print(f'{movie} 예상 최종관객수:', result)

#model_Train() #학습 시 사용
model_Run() #테스트 시 사용



