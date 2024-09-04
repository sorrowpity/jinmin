import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from re import sub
from decimal import Decimal
from plotnine import *
import matplotlib as mpl
import scipy
from plotnine import *

# 设置matplotlib的全局参数
mpl.rcParams['figure.figsize'] = [15,9]  # 设置图形的大小
mpl.style.use('ggplot')  # 使用ggplot风格
mpl.rcParams['font.family']= "STKaiti"  # 设置字体为楷体
mpl.rcParams['axes.unicode_minus']=False  # 确保负号显示正常

# 读取数据
listings = pd.read_csv("F://software_learn//python//pythonProject//listings.csv")
reviews = pd.read_csv("F://software_learn//python//pythonProject//reviews.csv")
calendar = pd.read_csv("F://software_learn//python//pythonProject//calendar.csv")

# 将价格中的逗号和美元符号去除，并转换为浮点数
price = [float(price[1:].replace(',', '')) for price in calendar.price]
print(listings.shape)  # 打印房源数据的维度
print(reviews.shape)  # 打印评论数据的维度
print(calendar.shape)  # 打印日历数据的维度
print(max(price))  # 打印最高价格
print(min(price))  # 打印最低价格
listings.columns  # 打印房源数据的列名

# 选择需要保留的列
columns_to_keep = ['id','listing_url','host_has_profile_pic','host_since','neighbourhood_cleansed', 'neighbourhood_group_cleansed',
                   'host_is_superhost','description',
                   'latitude', 'longitude','property_type', 'room_type', 'accommodates', 'bathrooms',
                   'bedrooms','beds', 'amenities', 'price',
                   'review_scores_rating','reviews_per_month', 'number_of_reviews',
                   'review_scores_accuracy','review_scores_cleanliness', 'review_scores_checkin',
                   'review_scores_communication','review_scores_location', 'review_scores_value',
                   'minimum_nights', 'host_response_rate',
                   'host_acceptance_rate', 'instant_bookable',
                   'availability_365']
listings = listings[columns_to_keep].set_index('id')  # 将id列设置为索引

# 将价格字符串转换为浮点数
listings.price=listings.price.str.replace('$','')
listings.price=listings.price.str.replace(',','').astype(float)

calendar.price = calendar.price.str.replace('$','')
calendar.price = calendar.price.str.replace(',','').astype(float)

# 将日期字符串转换为datetime类型
reviews['date'] = pd.to_datetime(reviews['date'], format='%Y-%m-%d')
calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')

# 打印缺失值大于300的列
columns_with_many_missing_values = listings.columns[listings.isna().sum() > 300]
print("Columns with more than 300 missing values:", columns_with_many_missing_values)

# 打印每列的缺失值数量
missing_values_count = listings.isna().sum()
print("Missing values count:\n", missing_values_count)

# 打印某些列的前几行
listings[['host_is_superhost',
         'instant_bookable', 'host_has_profile_pic']].head()

# 将某些列的字符串值转换为整数
for column in ['host_is_superhost',
               'instant_bookable', 'host_has_profile_pic']:
    listings[column] = listings[column].map({'f':0,'t':1})

# 打印价格列的缺失值数量
listings[['price']].isna().sum()

# 打印价格列的描述性统计
print(listings['price'].describe())

# 绘制价格的对数直方图和核密度估计
sns.histplot(np.log(listings.price), kde=True)
fig = plt.figure()
res = scipy.stats.probplot(np.log(listings.price), plot=plt)
plt.show()

# 绘制房源类型的条形图
listings['property_type'].value_counts()[:20].plot(kind='bar')
plt.figure()
sns.countplot(x='room_type',data=listings)
plt.show()

# 处理neighbourhood_cleansed列，只保留第一个词
listings['neighbourhood_cleansed'] = [x.split(' ')[0] for x in listings['neighbourhood_cleansed']]
# 绘制处理后的neighbourhood_cleansed列的条形图
plt.figure()
listings['neighbourhood_cleansed'].value_counts().plot(kind='bar')
plt.show()

# 导入geopandas库
import geopandas as gpd

# 读取GeoJSON文件
NewYork_gd = gpd.GeoDataFrame.from_file("D:/pycharm/NewYork.geojson")

# 打印GeoDataFrame的列名
print("Columns in GeoDataFrame:", NewYork_gd.columns)

# 计算每个区域的房源数量
counts = listings['neighbourhood_cleansed'].value_counts()
to_add = []

# 假设'name'是包含区域名称的字段
for i in range(len(NewYork_gd.index)):
    neighbourhood = NewYork_gd.iloc[i].name
    to_add.append(counts.get(neighbourhood, 0))

NewYork_gd['counts'] = to_add

# 绘制地理图
NewYork_gd.plot(column='counts', cmap='viridis')
plt.show()

# 导入folium库
import folium

# 创建GeoDataFrame，包含房源的经纬度信息
locations = gpd.GeoDataFrame(geometry=gpd.points_from_xy(listings.longitude, listings.latitude))
locations.crs = {'init' :'epsg:4326'}

# 创建folium地图
NewYork_map = folium.Map(
    location=[42.6520, -73.7562],
    zoom_start=12)  # 设置初始缩放级别

# 将房源的经纬度添加到地图上
points = folium.features.GeoJson(locations[:100])
NewYork_map.add_child(points)
NewYork_map.save('airbnb100.html')

# 计算评论数量
review_counts = reviews['date'].value_counts()
review_counts=review_counts[review_counts.index>'2015-08-01']
review_counts.sort_index(ascending=True, inplace=True)

# 绘制评论数量的折线图
plt.plot(review_counts, 'o', alpha=0.3)
reviews['date'].value_counts()
rolling_mean = review_counts.rolling(window=30).mean()
plt.plot(rolling_mean,lw=3)
plt.show()