#!/usr/bin/python3

# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import geopandas as gpd
from shapely.geometry import Point
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


listings = pd.read_csv('listings.csv')
reviews = pd.read_csv('reviews.csv')

calendar = pd.read_csv('calendar.csv', low_memory=False)

# 2. 数据清洗
# 清洗listings数据集
listings = listings.dropna(subset=['price', 'latitude', 'longitude'])
listings['price'] = listings['price'].str.replace('$', '')  # 移除美元符号
listings['price'] = listings['price'].str.replace(',', '')  # 移除逗号
listings['price'] = listings['price'].astype(float)  # 转换为浮点数

# 转换minimum_nights和maximum_nights为整数，如果存在的话
listings['minimum_nights'] = pd.to_numeric(listings['minimum_nights'], errors='coerce').astype(int)
listings['maximum_nights'] = pd.to_numeric(listings['maximum_nights'], errors='coerce').astype(int)



# 清洗reviews数据集
reviews = reviews.dropna(subset=['listing_id', 'id', 'date', 'reviewer_id', 'reviewer_name', 'comments'])
reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')

# 清洗calendar数据集
calendar = calendar.dropna(subset=['listing_id', 'date', 'available', 'price'])
calendar['date'] = pd.to_datetime(calendar['date'], errors='coerce')

# 移除价格中的美元符号和逗号，然后转换为浮点数
calendar['price'] = calendar['price'].str.replace('$', '')  # 移除美元符号
calendar['price'] = calendar['price'].str.replace(',', '')  # 移除逗号
calendar['price'] = calendar['price'].astype(float)  # 转换为浮点数

calendar = calendar[calendar['available'] == 't']  # 只保留available为t的记录

calendar['adjusted_price']=calendar['adjusted_price'].str.replace('$', '')
calendar['adjusted_price']=calendar['adjusted_price'].str.replace(',', '')
calendar['adjusted_price']=calendar['adjusted_price'].astype(float)


# 描述性统计
print(listings[['price', 'minimum_nights', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds']].describe())

#价格分布
sns.histplot(listings['price'], bins=50, kde=True)
plt.title('Distribution of Listing Prices')
plt.xlabel('Price ($)')
plt.ylabel('Number of Listings')
plt.show()

# 价格与住宿人数关系
sns.scatterplot(x='accommodates', y='price', data=listings)
plt.title('Price vs Accommodates')
plt.xlabel('Accommodates')
plt.ylabel('Price ($)')
plt.show()


# 选择特征
features = ['latitude', 'longitude', 'price', 'accommodates', 'bedrooms', 'beds']
X = listings[features]

# 检查是否有缺失值
if X.isnull().values.any():
    # 实例化一个 SimpleImputer 对象，使用均值来填补缺失值
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(X)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练KMeans模型
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 将聚类结果添加到 listings DataFrame
listings['cluster'] = clusters

# 可视化聚类结果
plt.figure(figsize=(10, 8))
sns.scatterplot(data=listings, x='longitude', y='latitude', hue='cluster', palette='viridis', s=10)
plt.title('KMeans Clustering of Listings')
plt.xlabel('Longitude') #经度
plt.ylabel('Latitude')  #纬度
plt.show()

##########################################################################3

# 描述性统计
print(reviews.describe())

reviews['year'] = reviews['date'].dt.year
reviews['month'] = reviews['date'].dt.month

# 计算每个月的评论数量
monthly_reviews = reviews.groupby(['year', 'month']).size().reset_index(name='count')

# 可视化评论数量随时间的变化
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_reviews, x='month', y='count', hue='year')
plt.title('Monthly Review Counts Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.show()


# 定义情感分析函数
def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

# 应用情感分析
reviews['sentiment'] = reviews['comments'].apply(sentiment_analysis)

# 可视化情感分析结果
plt.figure(figsize=(10, 6))
sns.histplot(reviews['sentiment'], bins=20, kde=True)
plt.title('Sentiment Analysis of Reviews')
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Reviews')
plt.show()


# 计算每个房源的评论数量
listing_reviews = reviews.groupby('listing_id').size().reset_index(name='review_count')

# # 可视化评论数量最多的房源
# plt.figure(figsize=(10, 6))
# sns.barplot(x='review_count', y='listing_id', data=listing_reviews.sort_values('review_count', ascending=False).head(20))
# plt.title('Top 20 Listings by Review Count')
# plt.xlabel('Review Count')
# plt.ylabel('Listing ID')
# plt.show()

# 描述性统计
print(calendar[['price', 'adjusted_price', 'minimum_nights', 'maximum_nights']].describe())


# 计算每个价格的出现次数
price_counts = calendar['price'].value_counts().sort_index()

# 绘制柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x=price_counts.index, y=price_counts.values, alpha=0.5)
plt.title('Distribution of Listing Prices')
plt.xlabel('Price ($)')
plt.ylabel('Number of Listings')
plt.show()



# 将 'available' 列中的 't' 转换为 1，'f' 转换为 0
calendar['available'] = calendar['available'].map({'t': 1, 'f': 0})

# 计算每个房源的总可用天数
calendar['available'] = calendar['available'].astype(int)  # 确保available列是整数类型
listing_availability = calendar.groupby('listing_id')['available'].sum().reset_index(name='total_available_days')





# 计算每个房源的可用天数的直方图
plt.figure(figsize=(10, 6))
sns.histplot(listing_availability['total_available_days'], bins=30, kde=False)
plt.title('Distribution of Total Available Days')
plt.xlabel('Total Available Days')
plt.ylabel('Number of Listings')
plt.show()




# 计算每个房源的平均价格
calendar['average_price'] = (calendar['price'] + calendar['adjusted_price']) / 2
listing_average_price = calendar.groupby('listing_id')['average_price'].mean().reset_index(name='average_price')
#
# 合并可用性和平均价格数据
listing_data = pd.merge(listing_availability, listing_average_price, on='listing_id')
#

# 对数据进行分组，可以按照总可用天数分组，计算每个组的平均价格
grouped_data = listing_data.groupby('total_available_days', as_index=False)['average_price'].mean()



# # 可视化价格与可用性的关系
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='total_available_days', y='average_price', data=listing_data)
# plt.title('Average Price vs Total Available Days')
# plt.xlabel('Total Available Days')
# plt.ylabel('Average Price ($)')
# plt.show()



















###废弃代码
# # 可视化房源的可用天数
# plt.figure(figsize=(10, 6))
# sns.barplot(x='listing_id', y='total_available_days', data=listing_availability.sort_values('total_available_days', ascending=False).head(20))
# plt.title('Top 20 Listings by Total Available Days')
# plt.xlabel('Listing ID')
# plt.ylabel('Total Available Days')
# plt.show()



# # 价格分布(以listing_id为y轴，price为x轴)
# sns.histplot(calendar['price'], bins=50, kde=True)
# plt.title('Distribution of Listing Prices')
# plt.bar(calendar['price'], calendar[''], alpha=0.5)
# plt.xlabel('Price ($)')
# plt.ylabel('Number of Listings')
# plt.show()



# # 3. 数据可视化
# sns.histplot(listings['price'], bins=25, kde=True)
# plt.title('Distribution of Listing Prices')
# plt.xlabel('Price ($)')
# plt.ylabel('Number of Listings')
# plt.show()
#
# # 检查评论数量
# print(reviews['listing_id'].value_counts().head())
#
# # 可视化评论数量
# reviews['count'] = 1
# review_counts = reviews.groupby('listing_id')['count'].sum().reset_index(name='review_count')
# sns.barplot(x='review_count', y='listing_id', data=review_counts.sort_values('review_count', ascending=False).head(20))
# plt.title('Top 20 Listings by Review Count')
# plt.xlabel('Review Count')
# plt.ylabel('Listing ID')
# plt.show()
#
# # 可视化评分分布
# sns.histplot(reviews['review_scores_rating'], bins=10, kde=True)
# plt.title('Distribution of Review Scores')
# plt.xlabel('Review Score')
# plt.ylabel('Number of Reviews')
# plt.show()