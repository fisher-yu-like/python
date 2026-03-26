import matplotlib.pyplot as plt
import pandas as pd
# 设置中文字体（Matplotlib默认不支持中文，这里是个小坑）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_data(csv_file):
    df = pd.read_csv(csv_file)

    # 创建一块画布，包含两个子图
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 评分分布直方图
    ax[0].hist(df['评分'], bins=10, color='skyblue', edgecolor='black')
    ax[0].set_title('豆瓣Top 250 评分分布')
    ax[0].set_xlabel('评分')
    ax[0].set_ylabel('电影数量')

    # 2. 年份分布柱状图（取前10个年份）
    year_counts = df['年份'].value_counts().sort_index().tail(15)
    year_counts.plot(kind='bar', ax=ax[1], color='salmon')
    ax[1].set_title('Top 250 电影年份分布 (近期)')
    ax[1].set_xlabel('年份')
    ax[1].set_ylabel('电影数量')

    plt.tight_layout()
    plt.show()

# 运行可视化
# visualize_data('douban_top250.csv')