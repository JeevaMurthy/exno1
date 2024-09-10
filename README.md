# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
REG NO : 212223230090
NAME : JEEVA K
```

```python
import pandas as pd
df=pd.read_csv("/content/SAMPLEIDS.csv")
df
```
![alt text](<Screenshot 2024-09-10 103959.png>)

```python
df.shape
```
![alt text](<Screenshot 2024-09-10 104015.png>)

```python
df.describe()
```
![alt text](<Screenshot 2024-09-10 104035.png>)

```python
df.info()
```
![alt text](<Screenshot 2024-09-10 104045.png>)

```python
df.head(4)
```
![alt text](<Screenshot 2024-09-10 104237.png>)

```python
df.tail(4)
```
![alt text](<Screenshot 2024-09-10 104320.png>)

```python
df.isna().sum()
```
![alt text](<Screenshot 2024-09-10 104334.png>)

```python
df.dropna(how='any').shape
```
![alt text](<Screenshot 2024-09-10 104343.png>)

```python
df.shape
```
![alt text](<Screenshot 2024-09-10 104351.png>)

```python
x=df.dropna(how='any')
x
```
![alt text](<Screenshot 2024-09-10 104403.png>)

```python
mn=df.TOTAL.mean()
mn
```
![alt text](<Screenshot 2024-09-10 104412.png>)

```python
df.TOTAL.fillna(mn,inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 104518.png>)

```python
df.isnull().sum()
```
![alt text](<Screenshot 2024-09-10 104706.png>)

```python
df.M1.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 104728.png>)

```python
df.isnull().sum()
```
![alt text](<Screenshot 2024-09-10 104741.png>)

```python
df.M2.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 104759.png>)

```python
df.isna().sum()
```
![alt text](<Screenshot 2024-09-10 104808.png>)

```python
df.M3.fillna(method='ffill',inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 104831.png>)

```python
df.isnull().sum()
```
![alt text](<Screenshot 2024-09-10 104840.png>)

```python
df.duplicated()
```
![alt text](<Screenshot 2024-09-10 104852.png>)

```python
df.drop_duplicates(inplace=True)
df
```
![alt text](<Screenshot 2024-09-10 104904.png>)

```python
df.duplicated()
```
![alt text](<Screenshot 2024-09-10 104917.png>)

```python
df['DOB']
```
![alt text](<Screenshot 2024-09-10 104925.png>)

```python
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](<Screenshot 2024-09-10 104933.png>)

```python
df.dropna(inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![alt text](<Screenshot 2024-09-10 104940.png>)

## 0UTLIERS DETECTION AND REMOVAL USING IQR

```python
import pandas as pd
import seaborn as sns
import numpy as np

```


```python
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
af
```
![alt text](<Screenshot 2024-09-10 104958.png>)

```python
sns.boxplot(data=af)
```
![alt text](<Screenshot 2024-09-10 105005.png>)

```python
sns.scatterplot(data=af)
```
![alt text](<Screenshot 2024-09-10 105012.png>)


```python
sns.boxenplot(data=af)
```
![alt text](<Screenshot 2024-09-10 105019.png>)


```python
q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1
iqr
```
![alt text](<Screenshot 2024-09-10 105026.png>)



```python
Q1=np.percentile(af,25)
Q3=np.percentile(af,75)
IQR = Q3 - Q1
IQR
```

![alt text](<Screenshot 2024-09-10 105032.png>)

```python
lower_bound = Q1 - 1.5 * IQR
lower_bound
```

![alt text](<Screenshot 2024-09-10 105046.png>)

```python
upper_bound = Q3 + 1.5 * IQR
upper_bound
```

![alt text](<Screenshot 2024-09-10 105052.png>)

```python
outliers = [x for x in age if x < lower_bound or x > upper_bound]
```


```python
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("LOWER BOUND:",lower_bound)
print("UPPER BOUND:",upper_bound)
print("OUTLIERS:",outliers)

```

![alt text](<Screenshot 2024-09-10 105102.png>)

```python
af=af[((af>=lower_bound)&(af<=upper_bound))]
af
```
![alt text](<Screenshot 2024-09-10 105110.png>)

```python
af.dropna()
```
![alt text](<Screenshot 2024-09-10 105217.png>)

```python
sns.boxplot(data=af)
```
![alt text](<Screenshot 2024-09-10 105224.png>)

```python
data = [1,2,2,2,3,1,1,15,2,2,2,3,1,1,2]
mean = np.mean(data)
std=np.std(data)
print('mean of the dataset:',mean)
print('std of dataset:',std)
```
![alt text](<Screenshot 2024-09-10 105232.png>)

```python
threshold = 3
outlier = []
for i in data:
  z= (i-mean)/std
  if z > threshold:
    outlier.append(i)
print("outlier in dataset:",outlier)
```
![alt text](<Screenshot 2024-09-10 105238.png>)

```python
data={'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
df=pd.DataFrame(data)
df

```
![alt text](<Screenshot 2024-09-10 105321.png>)

```python
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
```


```python
z=np.abs(stats.zscore(df))
print(df[z['weight']>3])
```
![alt text](<Screenshot 2024-09-10 105335.png>)
# Result
Thus we have cleaned the data and removed the outliers by detection using IQR
 and Z-score method.
