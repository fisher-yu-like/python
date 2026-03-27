import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

def scrape_douban():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://movie.douban.com/top250",
        "Cookie": "bid=KDEWOo2sKOc; _pk_id.100001.4cf6=6782dc136dcfb21e.1774572503.; _pk_ses.100001.4cf6=1; ap_v=0,6.0; __utma=30149280.403158839.1774572504.1774572504.1774572504.1; __utmb=30149280.0.10.1774572504; __utmc=30149280; __utmz=30149280.1774572504.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utma=223695111.1612193908.1774572504.1774572504.1774572504.1; __utmb=223695111.0.10.1774572504; __utmc=223695111; __utmz=223695111.1774572504.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none)"
    }

    movie_list = []
    print("🚀 开始爬取豆瓣 Top 250...")

    for i in range(0, 250, 25):
        url = f"https://movie.douban.com/top250?start={i}"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"❌ 第 {i // 25 + 1} 页访问受阻，状态码: {response.status_code}")
                break

            # --- 修正点 1：确保 soup 在正确的作用域 ---
            soup = BeautifulSoup(response.text, "html.parser")
            movie_items = soup.find_all("div", class_="item")

            if not movie_items:
                print(f"⚠️ 第 {i // 25 + 1} 页没有找到内容，检查 Cookie 是否过期。")
                break

            for item in movie_items:
                try:
                    # 1. 标题 (只取第一个 title)
                    title_tag = item.find("span", class_="title")
                    title = title_tag.get_text(strip=True) if title_tag else "未知标题"

                    # 2. 年份 (修正点 2：豆瓣的年份通常在 class 为空或特定结构的 p 标签中)
                    # 我们寻找包含“导演”或“/”文本的那个 p
                    info_tag = item.find("p", class_="")
                    year = "未知年份"
                    if info_tag:
                        # 使用 get_text() 获取所有文本，包括换行符内的数字
                        info_text = info_tag.get_text(strip=True)
                        # 正则匹配：查找 4 位连续数字
                        year_match = re.search(r'\d{4}', info_text)
                        if year_match:
                            year = year_match.group()

                    # 3. 评分
                    rating_tag = item.find("span", class_="rating_num")
                    rating = rating_tag.get_text(strip=True) if rating_tag else "0.0"

                    # 4. 短评 (防御性提取)
                    quote_tag = item.find("span", class_="quote")
                    quote = quote_tag.get_text(strip=True) if quote_tag else "暂无短评"

                    movie_list.append([title, year, rating, quote])

                except Exception as e:
                    print(f"解析单个电影出错: {e}")
                    continue

            print(f"✅ 已完成第 {i // 25 + 1} 页 (累计: {len(movie_list)})")
            # 休息一下，防止被封
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            print(f"请求第 {i // 25 + 1} 页时发生崩溃: {e}")
            break

    # --- 最终保存 ---
    if movie_list:
        df = pd.DataFrame(movie_list, columns=['电影名', '年份', '评分', '短评'])
        df.to_csv('douban_top250.csv', index=False, encoding='utf-8-sig')
        print(f"\n🎉 抓取成功！共 {len(movie_list)} 条数据已存入 'douban_top250.csv'")
        return df
    else:
        print("\n😭 任务结束，但没有抓取到任何数据。")

# 执行
scrape_douban()