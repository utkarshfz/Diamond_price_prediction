import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('diamonds.csv')
df.head()
df['volume'] = df['x']*df['y']*df['z']  


df.info()


print(df['price'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['price']);
plt.title('Diamond price dist')
plt.show()




# First subplot showing the diamond carat weight distribution
plt.subplot(221)
plt.hist(df['carat'],color='g')
plt.xlabel('Carat')
plt.ylabel('Frequency')
plt.title('Diamond wt distribution')
# Second subplot showing the diamond depth distribution
plt.subplot(222)
plt.hist(df['depth'],color='b')
plt.xlabel('Depth (%)')
plt.ylabel('Frequency')
plt.title('Diamond depth distribution')


# Third subplot showing the diamond price distribution
plt.subplot(223)
plt.hist(df['price'],color='r')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Diamond Price distribution')

# Fourth subplot showing the diamond volume distribution
plt.subplot(224)
plt.hist(df['volume'],bins=20,color='m')
plt.xlabel('Volume(cc)')
plt.ylabel('Frequency')
plt.title('Diamond Volume distribution')
plt.show()


data = df.select_dtypes(include = ['float64', 'int64'])


sns.pairplot(data=data,
            x_vars=data.columns[0:9],
            y_vars=['price'])

plt.show()


data.hist(figsize=(10, 10));
plt.show()



corr = data.corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))

correlation = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
sns.heatmap(correlation)

plt.show()
