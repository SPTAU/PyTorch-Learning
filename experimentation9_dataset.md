"""
Filename: experimentation9_dataset.ipynb
Author: SPTAU
"""

对数据集进行处理

导入库


```python
import os

import numpy as np
import pandas as pd
```

设置 dataset 地址


```python
TRAIN_PATH = "./dataset/otto-group-product-classification-challenge/train.csv"
TEST_PATH = "./dataset/otto-group-product-classification-challenge/test.csv"
SAMPLE_SUBMISSION_PATH = "./dataset/otto-group-product-classification-challenge/sampleSubmission.csv"
PROCESSED_TRAIN_PATH = "./dataset/otto-group-product-classification-challenge/processed_train.csv"
```

读取 training dataset


```python
train_data = pd.read_csv(TRAIN_PATH, index_col=0)
```

显示 training dataset 信息


```python
train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 61878 entries, 1 to 61878
    Data columns (total 94 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   feat_1   61878 non-null  int64 
     1   feat_2   61878 non-null  int64 
     2   feat_3   61878 non-null  int64 
     3   feat_4   61878 non-null  int64 
     4   feat_5   61878 non-null  int64 
     5   feat_6   61878 non-null  int64 
     6   feat_7   61878 non-null  int64 
     7   feat_8   61878 non-null  int64 
     8   feat_9   61878 non-null  int64 
     9   feat_10  61878 non-null  int64 
     10  feat_11  61878 non-null  int64 
     11  feat_12  61878 non-null  int64 
     12  feat_13  61878 non-null  int64 
     13  feat_14  61878 non-null  int64 
     14  feat_15  61878 non-null  int64 
     15  feat_16  61878 non-null  int64 
     16  feat_17  61878 non-null  int64 
     17  feat_18  61878 non-null  int64 
     18  feat_19  61878 non-null  int64 
     19  feat_20  61878 non-null  int64 
     20  feat_21  61878 non-null  int64 
     21  feat_22  61878 non-null  int64 
     22  feat_23  61878 non-null  int64 
     23  feat_24  61878 non-null  int64 
     24  feat_25  61878 non-null  int64 
     25  feat_26  61878 non-null  int64 
     26  feat_27  61878 non-null  int64 
     27  feat_28  61878 non-null  int64 
     28  feat_29  61878 non-null  int64 
     29  feat_30  61878 non-null  int64 
     30  feat_31  61878 non-null  int64 
     31  feat_32  61878 non-null  int64 
     32  feat_33  61878 non-null  int64 
     33  feat_34  61878 non-null  int64 
     34  feat_35  61878 non-null  int64 
     35  feat_36  61878 non-null  int64 
     36  feat_37  61878 non-null  int64 
     37  feat_38  61878 non-null  int64 
     38  feat_39  61878 non-null  int64 
     39  feat_40  61878 non-null  int64 
     40  feat_41  61878 non-null  int64 
     41  feat_42  61878 non-null  int64 
     42  feat_43  61878 non-null  int64 
     43  feat_44  61878 non-null  int64 
     44  feat_45  61878 non-null  int64 
     45  feat_46  61878 non-null  int64 
     46  feat_47  61878 non-null  int64 
     47  feat_48  61878 non-null  int64 
     48  feat_49  61878 non-null  int64 
     49  feat_50  61878 non-null  int64 
     50  feat_51  61878 non-null  int64 
     51  feat_52  61878 non-null  int64 
     52  feat_53  61878 non-null  int64 
     53  feat_54  61878 non-null  int64 
     54  feat_55  61878 non-null  int64 
     55  feat_56  61878 non-null  int64 
     56  feat_57  61878 non-null  int64 
     57  feat_58  61878 non-null  int64 
     58  feat_59  61878 non-null  int64 
     59  feat_60  61878 non-null  int64 
     60  feat_61  61878 non-null  int64 
     61  feat_62  61878 non-null  int64 
     62  feat_63  61878 non-null  int64 
     63  feat_64  61878 non-null  int64 
     64  feat_65  61878 non-null  int64 
     65  feat_66  61878 non-null  int64 
     66  feat_67  61878 non-null  int64 
     67  feat_68  61878 non-null  int64 
     68  feat_69  61878 non-null  int64 
     69  feat_70  61878 non-null  int64 
     70  feat_71  61878 non-null  int64 
     71  feat_72  61878 non-null  int64 
     72  feat_73  61878 non-null  int64 
     73  feat_74  61878 non-null  int64 
     74  feat_75  61878 non-null  int64 
     75  feat_76  61878 non-null  int64 
     76  feat_77  61878 non-null  int64 
     77  feat_78  61878 non-null  int64 
     78  feat_79  61878 non-null  int64 
     79  feat_80  61878 non-null  int64 
     80  feat_81  61878 non-null  int64 
     81  feat_82  61878 non-null  int64 
     82  feat_83  61878 non-null  int64 
     83  feat_84  61878 non-null  int64 
     84  feat_85  61878 non-null  int64 
     85  feat_86  61878 non-null  int64 
     86  feat_87  61878 non-null  int64 
     87  feat_88  61878 non-null  int64 
     88  feat_89  61878 non-null  int64 
     89  feat_90  61878 non-null  int64 
     90  feat_91  61878 non-null  int64 
     91  feat_92  61878 non-null  int64 
     92  feat_93  61878 non-null  int64 
     93  target   61878 non-null  object
    dtypes: int64(93), object(1)
    memory usage: 44.8+ MB
    

查看 training dataset 前几行


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
      <th>feat_4</th>
      <th>feat_5</th>
      <th>feat_6</th>
      <th>feat_7</th>
      <th>feat_8</th>
      <th>feat_9</th>
      <th>feat_10</th>
      <th>...</th>
      <th>feat_85</th>
      <th>feat_86</th>
      <th>feat_87</th>
      <th>feat_88</th>
      <th>feat_89</th>
      <th>feat_90</th>
      <th>feat_91</th>
      <th>feat_92</th>
      <th>feat_93</th>
      <th>target</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Class_1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 94 columns</p>
</div>



查找是否存在缺失值

pandas.isnull()
可以以布尔类型返回各行各列是否存在缺失
加上 sum() 函数可以统计各列的缺失情况

若尘公子 - [#有空学04# pandas缺失数据查询](https://zhuanlan.zhihu.com/p/158684561)
pandas - [pandas.isnull](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isnull.html)


```python
train_data.isnull().sum()
```




    feat_1     0
    feat_2     0
    feat_3     0
    feat_4     0
    feat_5     0
              ..
    feat_90    0
    feat_91    0
    feat_92    0
    feat_93    0
    target     0
    Length: 94, dtype: int64



读取 testing dataset


```python
test_data = pd.read_csv(TEST_PATH, index_col=0)
```

显示 testing dataset 信息


```python
test_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 144368 entries, 1 to 144368
    Data columns (total 93 columns):
     #   Column   Non-Null Count   Dtype
    ---  ------   --------------   -----
     0   feat_1   144368 non-null  int64
     1   feat_2   144368 non-null  int64
     2   feat_3   144368 non-null  int64
     3   feat_4   144368 non-null  int64
     4   feat_5   144368 non-null  int64
     5   feat_6   144368 non-null  int64
     6   feat_7   144368 non-null  int64
     7   feat_8   144368 non-null  int64
     8   feat_9   144368 non-null  int64
     9   feat_10  144368 non-null  int64
     10  feat_11  144368 non-null  int64
     11  feat_12  144368 non-null  int64
     12  feat_13  144368 non-null  int64
     13  feat_14  144368 non-null  int64
     14  feat_15  144368 non-null  int64
     15  feat_16  144368 non-null  int64
     16  feat_17  144368 non-null  int64
     17  feat_18  144368 non-null  int64
     18  feat_19  144368 non-null  int64
     19  feat_20  144368 non-null  int64
     20  feat_21  144368 non-null  int64
     21  feat_22  144368 non-null  int64
     22  feat_23  144368 non-null  int64
     23  feat_24  144368 non-null  int64
     24  feat_25  144368 non-null  int64
     25  feat_26  144368 non-null  int64
     26  feat_27  144368 non-null  int64
     27  feat_28  144368 non-null  int64
     28  feat_29  144368 non-null  int64
     29  feat_30  144368 non-null  int64
     30  feat_31  144368 non-null  int64
     31  feat_32  144368 non-null  int64
     32  feat_33  144368 non-null  int64
     33  feat_34  144368 non-null  int64
     34  feat_35  144368 non-null  int64
     35  feat_36  144368 non-null  int64
     36  feat_37  144368 non-null  int64
     37  feat_38  144368 non-null  int64
     38  feat_39  144368 non-null  int64
     39  feat_40  144368 non-null  int64
     40  feat_41  144368 non-null  int64
     41  feat_42  144368 non-null  int64
     42  feat_43  144368 non-null  int64
     43  feat_44  144368 non-null  int64
     44  feat_45  144368 non-null  int64
     45  feat_46  144368 non-null  int64
     46  feat_47  144368 non-null  int64
     47  feat_48  144368 non-null  int64
     48  feat_49  144368 non-null  int64
     49  feat_50  144368 non-null  int64
     50  feat_51  144368 non-null  int64
     51  feat_52  144368 non-null  int64
     52  feat_53  144368 non-null  int64
     53  feat_54  144368 non-null  int64
     54  feat_55  144368 non-null  int64
     55  feat_56  144368 non-null  int64
     56  feat_57  144368 non-null  int64
     57  feat_58  144368 non-null  int64
     58  feat_59  144368 non-null  int64
     59  feat_60  144368 non-null  int64
     60  feat_61  144368 non-null  int64
     61  feat_62  144368 non-null  int64
     62  feat_63  144368 non-null  int64
     63  feat_64  144368 non-null  int64
     64  feat_65  144368 non-null  int64
     65  feat_66  144368 non-null  int64
     66  feat_67  144368 non-null  int64
     67  feat_68  144368 non-null  int64
     68  feat_69  144368 non-null  int64
     69  feat_70  144368 non-null  int64
     70  feat_71  144368 non-null  int64
     71  feat_72  144368 non-null  int64
     72  feat_73  144368 non-null  int64
     73  feat_74  144368 non-null  int64
     74  feat_75  144368 non-null  int64
     75  feat_76  144368 non-null  int64
     76  feat_77  144368 non-null  int64
     77  feat_78  144368 non-null  int64
     78  feat_79  144368 non-null  int64
     79  feat_80  144368 non-null  int64
     80  feat_81  144368 non-null  int64
     81  feat_82  144368 non-null  int64
     82  feat_83  144368 non-null  int64
     83  feat_84  144368 non-null  int64
     84  feat_85  144368 non-null  int64
     85  feat_86  144368 non-null  int64
     86  feat_87  144368 non-null  int64
     87  feat_88  144368 non-null  int64
     88  feat_89  144368 non-null  int64
     89  feat_90  144368 non-null  int64
     90  feat_91  144368 non-null  int64
     91  feat_92  144368 non-null  int64
     92  feat_93  144368 non-null  int64
    dtypes: int64(93)
    memory usage: 103.5 MB
    

查看 testing dataset 前几行


```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
      <th>feat_4</th>
      <th>feat_5</th>
      <th>feat_6</th>
      <th>feat_7</th>
      <th>feat_8</th>
      <th>feat_9</th>
      <th>feat_10</th>
      <th>...</th>
      <th>feat_84</th>
      <th>feat_85</th>
      <th>feat_86</th>
      <th>feat_87</th>
      <th>feat_88</th>
      <th>feat_89</th>
      <th>feat_90</th>
      <th>feat_91</th>
      <th>feat_92</th>
      <th>feat_93</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>14</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 93 columns</p>
</div>



查找是否存在缺失值


```python
test_data.isnull().sum()
```




    feat_1     0
    feat_2     0
    feat_3     0
    feat_4     0
    feat_5     0
              ..
    feat_89    0
    feat_90    0
    feat_91    0
    feat_92    0
    feat_93    0
    Length: 93, dtype: int64



统计 target 列中的类别和数量

快乐的皮卡丘呦呦 - [Pandas中查看列中数据的种类及个数](https://www.bbsmax.com/A/mo5k0wkndw/)
pandas - [pandas.DataFrame.value_counts](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.value_counts.html)


```python
train_data['target'].value_counts()
```




    Class_2    16122
    Class_6    14135
    Class_8     8464
    Class_3     8004
    Class_9     4955
    Class_7     2839
    Class_5     2739
    Class_4     2691
    Class_1     1929
    Name: target, dtype: int64



根据上面的信息，可以看到 training dataset 的特征为 int ，标签为 Classs_1 ~ Class_9 的字符串

现在需要将标签转换为 one-hot 格式

ChaoFeiLi - [操作pandas某一列实现one-hot](https://blog.csdn.net/ChaoFeiLi/article/details/115345237)
pandas - [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)

pd.get_dummies() 会转换非数值型数据，即在该数据集中不指定 columns 参数，也只会转换 target 一列
pandas - [pandas.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)


```python
train_data = pd.get_dummies(train_data)
# train_data = pd.get_dummies(train_data, columns=['target'])
```

确认处理结果


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
      <th>feat_4</th>
      <th>feat_5</th>
      <th>feat_6</th>
      <th>feat_7</th>
      <th>feat_8</th>
      <th>feat_9</th>
      <th>feat_10</th>
      <th>...</th>
      <th>feat_93</th>
      <th>target_Class_1</th>
      <th>target_Class_2</th>
      <th>target_Class_3</th>
      <th>target_Class_4</th>
      <th>target_Class_5</th>
      <th>target_Class_6</th>
      <th>target_Class_7</th>
      <th>target_Class_8</th>
      <th>target_Class_9</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 102 columns</p>
</div>




```python
train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 61878 entries, 1 to 61878
    Columns: 102 entries, feat_1 to target_Class_9
    dtypes: int64(93), uint8(9)
    memory usage: 44.9 MB
    

pandas - [pandas.DataFrame.notnull](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.notnull.html)


```python
train_data.notnull().sum()
```




    feat_1            61878
    feat_2            61878
    feat_3            61878
    feat_4            61878
    feat_5            61878
                      ...  
    target_Class_5    61878
    target_Class_6    61878
    target_Class_7    61878
    target_Class_8    61878
    target_Class_9    61878
    Length: 102, dtype: int64



写入到 CSV 文件中


```python
train_data.to_csv(PROCESSED_TRAIN_PATH, index=False)
```
