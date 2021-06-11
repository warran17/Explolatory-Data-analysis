#!/usr/bin/env python
# coding: utf-8

# # Research on car sales ads
# 
# Hundreds of free advertisements for vehicles are published on your site every day. We need to study data collected over the last few years and determine which factors influence the price of a vehicle.

# ### Step 1. Open the data file and study the general information. 

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df= pd.read_csv('/datasets/vehicles_us.csv')
print(df.info())


# ### Conclusion

# Data has 51525 rows and 13 different columns. Among them some columns have missing values as they have less number of non-null values compared to number of rows.

# 

# ### Step 2. Data preprocessing

# In[2]:


df.head(15)


# In[3]:


print(df[df.duplicated()])


# >There are not duplicated data

# In[4]:


print(df[df['odometer'].isnull()].groupby('model')['model_year'].value_counts())


# In[5]:


print(df[df['model_year'].isnull()].groupby('model')['type'].value_counts())


# >There are no duplicated data

# In[6]:


print(df[df['cylinders'].isnull()].groupby('model')['type'].value_counts())


# In[7]:


print(df[df['paint_color'].isnull()].groupby('model')['model_year'].value_counts())


# In[8]:


print(df[df['odometer'].isnull()].groupby('model_year')['price'].value_counts())


# In[9]:


print(df[df['odometer'].isnull()].groupby('model')['model_year'].value_counts())


# In[10]:


#print(df[df['odometer'].isnull()].groupby('model')['model_year'].value_counts())
df_null=df[df['odometer'].isnull()]
#print(df_null['model_year'].value_counts()/df['model_year'].value_counts())
print(df_null['model_year'].value_counts().head(10))


# In[11]:


print(df['model_year'].value_counts().head(10))


# In[12]:


print(df[df['paint_color'].isnull()].groupby('condition')['model_year'].value_counts())


# <h3>Missing values seems random in nature except in _'is_4wd'_ column </h3>
# 
# >As the missing values are distributed in all model and there do not seem to have any relation to any specific value in other column
# 
# >One obivious thing could be a approach taken by data entry person. Data entry person may not type the value in another row if it has same value as in earlier row.
#     The chances of having such approach is possible due to the fact that the missing values are not the different value than the existing value.
#     As model_year, cylinders and paint_color have limited set of data from where missing value should come. In such case missing value
#     can be filled by assigning *ffill* method in *fillna* syntax.
#     
#     But there is no strong evidence of missing value is due to data entry person this idea is not applied in this project
#     
# >In case of odometer, the missing values are distributed along all the model. It should be completly random.

# In[13]:


df['is_4wd']=df['is_4wd'].fillna(value=0) 


# In[14]:


print(df['cylinders'].value_counts())


# In[15]:


print(df.groupby('model')['cylinders'].mean())


# In[16]:


df['cylinders'] = df['cylinders'].fillna(df.groupby(['model','type'])['cylinders'].transform('median'))
df.info()


# In[17]:


df['cylinders'] = df['cylinders'].fillna(df.groupby(['model','model_year'])['cylinders'].transform('median'))
df.info()


# In[18]:


df['model_year'] = df['model_year'].fillna(df.groupby('model')['model_year'].transform('median'))
print(df)


# In[19]:


df['odometer'] = df['odometer'].fillna(df.groupby(['model','model_year'])['odometer'].transform('mean'))
df['odometer'] = df['odometer'].fillna(df.groupby(['model','type'])['odometer'].transform('mean'))
df['odometer'] = df['odometer'].fillna(df['odometer'].mean())
print(df.info())


# In[20]:


df['date_posted'] = pd.to_datetime(df['date_posted'], format='%Y-%m-%d')
df.info()


# In[21]:



df['day_of_week']=df['date_posted'].dt.weekday
df['month']=df['date_posted'].dt.month
df['year']=df['date_posted'].dt.year
df.info()


# <h3>In data processing part, first missing values are observed and filled in.</h3>
# 
# In case of 'is_4wd' column the missing value were False boolean value. So they filled with 0.
# Then, missing value in column 'cyinders' is filled by median value of each vehicle 'type'. 
# 
# After, that remaing missing value are filled with median value of missing year. Median is taken because, the number of cylinders
# will have integer number and there are only few possible values.
# 
# Then, missing 'model_year' values are filled based on 'model'. The median value is taken since, year cannot have floating value.
# The missing values in 'odometer' are filled by mean value based on 'model_year' for each model. Remaining missing values are filled by mean value based on 
# 'type' of each model. Rest are filled with overall mean value of the column.
# 
# The 'date_posted' column had object datatype which is changed to datetime datatype so that panda can read and process it as a
# date rather than a simple string. Then, columns for year, month and week of day are created and assigned their respective values from 'date_posted'
# column.
# 
# 

# ### Step 3. Make calculations and add them to the table

# In[22]:


df['vehicles_age']=df['year']-df['model_year'].astype(int)


# In[23]:


df['milage_per_year']=df['odometer']/df['vehicles_age']


# In[24]:



df.loc[df['condition']=='new','condition']=int(5)
df.loc[df['condition']=='like new','condition']=int(4)
df.loc[df['condition']=='excellent','condition']=int(3)
df.loc[df['condition']=='good','condition']=int(2)
df.loc[df['condition']=='fair','condition']=int(1)
df.loc[df['condition']=='salvage','condition']=int(0)
df['condition']=df['condition'].astype(int)
df.info()


# In this section vehicles age and milage per year are determined. With these values two new columns are created. Likewise, condition
# column replaced by numeric value based on condition.

# ### Step 4. Carry out exploratory data analysis

# In[25]:


df.hist('price', bins=100) 
plt.title('Histogram including outlier')
plt.xlabel('Price')
plt.ylabel('frequency')


# > In this figure, histogram including outlier do not have proper utilization of screen. Though it seems price have gausian distibution with longer tail in right side, values are not clear. It is difficult to say the price range of cars which has altogether 2000 number of car of that price.

# In[26]:


print(df['price'].max())
print(df['price'].min())


# In[27]:


df.hist('vehicles_age', bins=30) 
plt.title('Histogram including outlier')
plt.xlabel('vehicles_age')
plt.ylabel('frequency')


# >In this figure, vehicles range are plotted upto more than 100 years. Due to this, enlarged distribution of vehicles age from zero to twenty years is not shown
# >Figure only shows most of the data in small portion of screen making difficult to find what is the exact age of most of the vehicles which belongs to peak and other bars.

# In[28]:


df.hist('milage_per_year', bins=25, range=(0,200000)) # range is set upto 200000 since outlier has infinite value
plt.title('Histogram for milage per year')
plt.xlabel('Milage per year')
plt.ylabel('frequency')


# In[29]:


df.hist('cylinders') 
plt.title('Histogram for cylinders')
plt.xlabel('No. of Cylinders')
plt.ylabel('frequency')


# >For number of cylinders picture seems clear since there is limited set of values possible for number of cylinders and there is not any outlier.
# 
# >Apparently, most of vehicles have eight cylinders followed by 17500 vehichles which have 6 cylinders. Also, vehicles having four cylinders are also are about 15000 in total.

# In[30]:


df.hist('condition') 
plt.title('Condition of vehicles')
plt.xlabel('condition')
plt.ylabel('frequency')


# >Since there are only five categories for condition of vehicles , there is not any outlier and number of vehicles in each category are apparent in figure.

# In[31]:


df.describe()


# In[32]:


df.boxplot('price')


# >The distribution of price is huge with outlier (about 40000), the box showing most of the vehicles concentration is not apparent to figure out. The single unit in scale made of 50000 which is not good for pictorial representation where most of vehicles price is below that value.

# In[33]:


df.boxplot('vehicles_age')


# > The figure is understandable that most of vehicles age ranges below 15 years. But with outliers upto more than 100 figure occupy 
# most of the space for outlier and make difficult to get exact number for data of interest.

# In[34]:


df.boxplot('milage_per_year')


# >The whisker seems symmetric and outliers are distributed upto about 300000 km per year. Figure is understandable but as above not easy to read values for the range of whisker.

# In[35]:


df.boxplot('condition')


# > As expalined in histogram, the graph is clear for condition. Whishker's range showing range of first quartile to third quartile is clear. 

# In[36]:


out_price=df['price'].quantile(0.75)+1.5*(df['price'].quantile(0.75)-df['price'].quantile(0.25))
out_age=df['vehicles_age'].quantile(0.75)+1.5*(df['vehicles_age'].quantile(0.75)-df['vehicles_age'].quantile(0.25))
out_milage=df['milage_per_year'].quantile(0.75)+1.5*(df['milage_per_year'].quantile(0.75)-df['milage_per_year'].quantile(0.25))
out_cylinders=df['cylinders'].quantile(0.75)+1.5*(df['cylinders'].quantile(0.75)-df['cylinders'].quantile(0.25))
out_condition=df['condition'].quantile(0.75)+1.5*(df['condition'].quantile(0.75)-df['condition'].quantile(0.25))
data_filtered=df
data_filtered= data_filtered[data_filtered['price']<out_price]
data_filtered= data_filtered[data_filtered['vehicles_age']<out_age]
data_filtered= data_filtered[data_filtered['milage_per_year']<out_milage]
data_filtered= data_filtered[data_filtered['cylinders']<out_cylinders]
data_filtered= data_filtered[data_filtered['condition']<out_condition]


# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 1:
#     
# Please see for details: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba 
# </div>

# In[37]:


data_filtered.info()


# In[38]:


data_filtered.hist('price',grid='True')
plt.title('Price of vehicles')
plt.xlabel('price')
plt.ylabel('number')


# > The figure is easy to read since there are no outlier, the figure drawn for place where data are concentrated. The 
# distribution is assymetric since the tail is longer in right side. There are maximum number of vehicles are in the price about 
# 5000. The second highest number of vehicles is in price ranging below 10000 and above 7000. After this, number of vehicles
# decreases with increasing price. Likewise, number of vehicles whose price is below 3500 are about 6000. From figure, the price has 
# assymetric gaussian distribution.

# In[39]:


data_filtered.hist('vehicles_age')
plt.title('Number of vehicles of different age')
plt.xlabel('vehicles age in year')
plt.ylabel('Total Number')


# >The vehicles age has almost uniform distribution upto 14 years with exception in between 3 to 7 years. Vehicles which aged in between 5 to 7 years has maximum number of advertisement which made up almost 8000 vehicles in total follwed by vehicles aging in between 3 to 5 at 7000 vehicles in total. As age approaches 15 years and more, number of vehicles decreases as age increases.

# In[40]:


data_filtered.hist('milage_per_year')
plt.title('Number of vehicles based on milage per year')
plt.xlabel('milage per year')
plt.ylabel('total number')


# >The milage of vehilces seems to be about 14000 in average and milage has gaussian distribution with slightly longer tail in right hand side of the mean.

# In[41]:


data_filtered.hist('cylinders')
plt.title('Vehicles Cylinders')
plt.xlabel('number of cylinders')
plt.ylabel('number of vehicles')


# > Similar to plot for unfiltered data, figure indicates the possible number of cylinder in each vehicles has limited set. Almost, all vehicles have either 4, 6 or 8 cylinders. Vehicles which has 6 cylinders made up maximum number of vehicles over 15000 followed by vehicles with 8 cylinders at about 15000.

# In[42]:


data_filtered.hist('condition')
plt.title('Condition of vehicles')
plt.xlabel('condition')
plt.ylabel('frequency')


# > In this figure, condition 5 is filtered out otherwise graph seems similar to figure for unfiltered data.

# In[43]:


data_filtered.hist('days_listed')
plt.title('Duration of advertisement')
plt.xlabel('days')
plt.ylabel('number')


# >Number of vehicles decreased with increasing number of days that advertisment listed showing exponentially decreasing trend.

# In[44]:


print(data_filtered['days_listed'].mean())
print(data_filtered['days_listed'].median())


# In[45]:


print(data_filtered['days_listed'].min())
print(data_filtered['days_listed'].max())


# In[46]:


min_displayed=data_filtered[data_filtered['days_listed']==data_filtered['days_listed'].min()]


# In[47]:


print(min_displayed.groupby('year')['month'].value_counts())


# In[48]:


print(min_displayed.groupby('year')['day_of_week'].value_counts())


# In[49]:



max_displayed=data_filtered[data_filtered['days_listed']==np.percentile(data_filtered['days_listed'],95)]
print(max_displayed.head())


# In[50]:


print(max_displayed.groupby('year')['month'].value_counts())


# In[51]:


print(max_displayed.groupby('year')['day_of_week'].value_counts())


# In[52]:


min_displayed.corr()


# In[53]:


max_displayed.corr()


# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 1:
#     
# Please take into account that it shows only linear dependecy between variables. Maybe this link will be interesting for you: https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/.
# </div>

# In[54]:


pivot_type=pd.pivot_table(data_filtered, index=['type'], values=['price'],
                    aggfunc=['mean','count'])
print(pivot_type)


# In[55]:


pd.pivot_table(data_filtered, index=['type'], values=['price']).sort_values(by='price', ascending=False).plot(kind='bar');
plt.title('Average price for different types')
plt.xlabel('model')
plt.ylabel('average price')


# **It is clear that, price depends on type of vehicles**

# In[56]:


pivot_type['count'].sort_values(by='price', ascending=False).plot(kind='bar');
plt.title('Vehicles of different type')
plt.xlabel('type')
plt.ylabel('total number')


# In[57]:


major=['SUV','coupe','mini-van ','sedan','truck']
data_filtered['type_group']=data_filtered['type'].where(data_filtered['type'].isin(major), 'Others')
print(data_filtered.head(15))
     


# In[58]:


data_filtered.plot(x='vehicles_age',y='price',kind='hexbin',gridsize=20, figsize=(8, 6), sharex=False, grid=True);


# In[59]:


print(data_filtered['vehicles_age'].corr(data_filtered['price']));


# In[60]:


data_filtered.plot(x='condition',y='price',kind='scatter', figsize=(8, 6), grid=True);


# In[61]:


print(data_filtered['condition'].corr(data_filtered['price']));


# In[62]:


data_filtered.plot(x='milage_per_year',y='price',kind='scatter', figsize=(8, 6), alpha=0.09,sharex=False, grid=True);


# In[63]:


print(data_filtered['milage_per_year'].corr(data_filtered['price']));


# In[64]:


data1=data_filtered[['price','transmission']]


# In[65]:


Q1 = data1['price'].quantile(0.25)
Q3 = data1['price'].quantile(0.75)
IQR = Q3 - Q1
#plt.ylim(-400, 1000)
data1.boxplot();
plt.hlines(y=[Q1,Q3,Q3+1.5*IQR], xmin=0.9, xmax=1.1, color='red');


# In[66]:


data2=data_filtered[['price','paint_color']]
Q1 = data2['price'].quantile(0.25)
Q3 = data2['price'].quantile(0.75)
IQR = Q3 - Q1
#plt.ylim(-400, 1000)
data1.boxplot();
plt.hlines(y=[Q1,Q3,Q3+1.5*IQR], xmin=0.9, xmax=1.1, color='red');


# In[67]:


data_filtered.corr()


# <h3> Conclusion: exploratory data analysis </h3>
#     
# >With outlier, the distinct price are not appear clearly and large range of data make single group and show a peak. On the other hand 
# after removing outliers, frequencies of small range appear and easier to understand the data. Distribution of data appear clearly. At the end,
# price seems to have some  relation with age of vehichle but not with milage per year and condition. In this condition, as long as vehicle is
# in good, excellent and 'like new' condition price could have high price too. 
# 
# >Likewise, number of cylinders, having 4wd or not and odometer reading also influence the price in some extent.

# ### Step 5. Overall conclusion

# <h4>The price of vechicle seem to have some relation with vehicles age. As age of vechicle increases price has decreasing trend. But age does not completly determine the price.
# Likewise, number of cylinders, having 4wd or not and odometer reading also influence the price in some extent. Furthermore, average price 
# was different type of vehicles in bar diagram drawn from pivot table suggest type of vehicles also the another determining factor of price of the vehicles.</h4>
# 
# >In this project, first duplicate rows are checked and missing values are analyzed for possible filling option. The column is_4wd has missing value for boolean
# '0'. So the missing values are filled with '0'. Other columns having missing value seems to be random and filled with median 
# values in these columns.**
# 
# * **Next, calulation is performed for vehicles age and mileage per year.**
#     > For this purpose, first 'object' type dates are converted to 'datetime'     datatype and stored values for year, month and 
#     day in different column
#     
#     > Calculated values are also saved in different column
#     
# * **exploratory data analysis is carried out**
#     > outliers are filtered out  
#     
#     > Different plot are drawn to see the relation of price with different factors
#     
#     > Likewise, exceptionally low and high number of days displayed found in 2018 and 2019. But, there is not any sign of any 
#     co-relation with other factors. Both 'days_listed' and 'year' column both seems to be independent of other parameters in the table
# 
# 
