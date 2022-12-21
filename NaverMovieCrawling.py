import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

def Crawling(result):
    movieCodes = [ 44885, 62560, 49008, 69989, 76348, 72363, 70254, 95873, 96327, 97857, 98438, 92064, 122527, 125459, 127398, 135874, 134898, 137326, 136315, 144330, 132623, 136900, 173123]

    for movieCode in movieCodes:
        url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code=%d'%movieCode
        response = urllib.request.urlopen(url)
        soupData = BeautifulSoup(response, 'html.parser')
        name = soupData.select_one('h3.h_movie a').text
        scoreCritic = soupData.select_one('div.special_score div.sc_view div.star_score em').text
        scoreNetizen = soupData.select_one('div.netizen_score div.sc_view div.star_score em').text
        print(name)
        print(scoreCritic)
        print(scoreNetizen)

        result.append([name] + [scoreCritic] + [scoreNetizen])


def CrawlingAll():
    result = [] # 결과 저장
    print('크롤링 시작')
    Crawling(result)
    print(result)
    resultTable = pd.DataFrame(result, columns=('제목', '평론가 평점', '네티즌 평점'))
    resultTable.to_csv("./CrawlingResult.csv", encoding="cp949", mode='w', index=True)
    del result[:]
    print('FINISHED')

if __name__ == '__main__':
    CrawlingAll()
