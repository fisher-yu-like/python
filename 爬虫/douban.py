import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random


def scrape_douban():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    movie_list = []

    print("开始爬取豆瓣 Top 250...")
    for i in range(0, 250, 25):  # 翻页逻辑：0, 25, 50...
        url = f"https://movie.douban.com/top250?start={i}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.find_all("div", class_="item")

            for item in items:
                # 核心解析逻辑
                title = item.find("span", class_="title").text  # 电影名
                info = item.find("p", class_="").text.strip()  # 导演、年份等原始文本
                year = info.split('\n')[-1].split('/')[0].strip()  # 提取年份
                rating = item.find("span", class_="rating_num").text  # 评分
                quote = item.find("span", class_="inq").text if item.find("span", class_="inq") else ""  # 短评

                movie_list.append([title, year, rating, quote])

            print(f"已完成第 {i // 25 + 1} 页爬取")
            time.sleep(random.uniform(1, 3))  # 随机延迟，模拟人类操作，防止被封IP

        except Exception as e:
            print(f"爬取失败: {e}")

    # 保存为 Excel/CSV 格式
    df = pd.DataFrame(movie_list, columns=['电影名', '年份', '评分', '短评'])
    df.to_csv('douban_top250.csv', index=False, encoding='utf-8-sig')
    print("数据已保存至 douban_top250.csv")
    return df

# 运行爬虫
df = scrape_douban()