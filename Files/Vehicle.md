# Определение стоимости автомобилей

Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 

### Заказчику важны:

- качество предсказания;
- скорость предсказания;
- время обучения.

### План работы

1. **Предобработка данных**
   
2. **Выбор модели**:
   - Использование моделей, таких как LightGBM, CatBoost, или LinearRegression.

   
3. **Оценка и настройка гиперпараметров**:
   - Оптимизация гиперпараметров с помощью GridSearchCV.

   
4. **Оценка качества модели**
   


### Признаки:

- **DateCrawled** — дата скачивания анкеты из базы
- **VehicleType** — тип автомобильного кузова
- **RegistrationYear** — год регистрации автомобиля
- **Gearbox** — тип коробки передач
- **Power** — мощность (л. с.)
- **Model** — модель автомобиля
- **Kilometer** — пробег (км)
- **RegistrationMonth** — месяц регистрации автомобиля
- **FuelType** — тип топлива
- **Brand** — марка автомобиля
- **Repaired** — была машина в ремонте или нет
- **DateCreated** — дата создания анкеты
- **NumberOfPictures** — количество фотографий автомобиля
- **PostalCode** — почтовый индекс владельца анкеты (пользователя)
- **LastSeen** — дата последней активности пользователя

Целевой признак:

- **Price** — цена (евро)



```python
# Отбор информативных признаков

cols = ['VehicleType', 'RegistrationYear', 'Gearbox', 'Power', 'Model',
        'Kilometer', 'FuelType', 'Brand', 'Repaired']
```

## Подготовка данных


```python
%pip install phik -q
%pip install category_encoders -q
%pip install catboost -q
%pip install lightgbm -q
```

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    


```python
import time
# импорт инструментов для анализа
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from phik import phik_matrix

# модели
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

# инструменты для обучения
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from category_encoders import TargetEncoder
from sklearn.preprocessing import (
    MinMaxScaler, 
    StandardScaler, 
    OneHotEncoder)
    
from sklearn.impute import SimpleImputer

# метрики
from sklearn.metrics import mean_squared_error, make_scorer

#Для оценки важности признаков
from sklearn.inspection import permutation_importance
```


```python
sns.set_style('darkgrid') # Установка стиля графиков по умолчанию

sns.set(rc={'figure.figsize':(10, 6)}) # Размер графиков по умолчанию
```


```python
path = r"C:\Users\79088\Documents\datasets\autos.csv"

df = pd.read_csv(path)
```


```python
df.head()
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
      <th>DateCrawled</th>
      <th>Price</th>
      <th>VehicleType</th>
      <th>RegistrationYear</th>
      <th>Gearbox</th>
      <th>Power</th>
      <th>Model</th>
      <th>Kilometer</th>
      <th>RegistrationMonth</th>
      <th>FuelType</th>
      <th>Brand</th>
      <th>Repaired</th>
      <th>DateCreated</th>
      <th>NumberOfPictures</th>
      <th>PostalCode</th>
      <th>LastSeen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-24 11:52:17</td>
      <td>480</td>
      <td>NaN</td>
      <td>1993</td>
      <td>manual</td>
      <td>0</td>
      <td>golf</td>
      <td>150000</td>
      <td>0</td>
      <td>petrol</td>
      <td>volkswagen</td>
      <td>NaN</td>
      <td>2016-03-24 00:00:00</td>
      <td>0</td>
      <td>70435</td>
      <td>2016-04-07 03:16:57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-24 10:58:45</td>
      <td>18300</td>
      <td>coupe</td>
      <td>2011</td>
      <td>manual</td>
      <td>190</td>
      <td>NaN</td>
      <td>125000</td>
      <td>5</td>
      <td>gasoline</td>
      <td>audi</td>
      <td>yes</td>
      <td>2016-03-24 00:00:00</td>
      <td>0</td>
      <td>66954</td>
      <td>2016-04-07 01:46:50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-14 12:52:21</td>
      <td>9800</td>
      <td>suv</td>
      <td>2004</td>
      <td>auto</td>
      <td>163</td>
      <td>grand</td>
      <td>125000</td>
      <td>8</td>
      <td>gasoline</td>
      <td>jeep</td>
      <td>NaN</td>
      <td>2016-03-14 00:00:00</td>
      <td>0</td>
      <td>90480</td>
      <td>2016-04-05 12:47:46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-17 16:54:04</td>
      <td>1500</td>
      <td>small</td>
      <td>2001</td>
      <td>manual</td>
      <td>75</td>
      <td>golf</td>
      <td>150000</td>
      <td>6</td>
      <td>petrol</td>
      <td>volkswagen</td>
      <td>no</td>
      <td>2016-03-17 00:00:00</td>
      <td>0</td>
      <td>91074</td>
      <td>2016-03-17 17:40:17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-31 17:25:20</td>
      <td>3600</td>
      <td>small</td>
      <td>2008</td>
      <td>manual</td>
      <td>69</td>
      <td>fabia</td>
      <td>90000</td>
      <td>7</td>
      <td>gasoline</td>
      <td>skoda</td>
      <td>no</td>
      <td>2016-03-31 00:00:00</td>
      <td>0</td>
      <td>60437</td>
      <td>2016-04-06 10:17:21</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 354369 entries, 0 to 354368
    Data columns (total 16 columns):
     #   Column             Non-Null Count   Dtype 
    ---  ------             --------------   ----- 
     0   DateCrawled        354369 non-null  object
     1   Price              354369 non-null  int64 
     2   VehicleType        316879 non-null  object
     3   RegistrationYear   354369 non-null  int64 
     4   Gearbox            334536 non-null  object
     5   Power              354369 non-null  int64 
     6   Model              334664 non-null  object
     7   Kilometer          354369 non-null  int64 
     8   RegistrationMonth  354369 non-null  int64 
     9   FuelType           321474 non-null  object
     10  Brand              354369 non-null  object
     11  Repaired           283215 non-null  object
     12  DateCreated        354369 non-null  object
     13  NumberOfPictures   354369 non-null  int64 
     14  PostalCode         354369 non-null  int64 
     15  LastSeen           354369 non-null  object
    dtypes: int64(7), object(9)
    memory usage: 43.3+ MB
    


```python
# Создание таблицы пустых значений

temp = dict()

for name in df.columns:
    count_null_values = sum(df[name].isna())
    
    proportion_null_values = count_null_values / df.shape[0]
    
    persent_null_values = round(proportion_null_values, 2) * 100
    
    temp[name] = persent_null_values
    
res = pd.DataFrame(data=temp.values(), index=temp.keys(), columns=['% null values'])
```


```python
res.sort_values(by='% null values', ascending=False)
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
      <th>% null values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Repaired</th>
      <td>20.0</td>
    </tr>
    <tr>
      <th>VehicleType</th>
      <td>11.0</td>
    </tr>
    <tr>
      <th>FuelType</th>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Gearbox</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Model</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>DateCrawled</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>RegistrationYear</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Power</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kilometer</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>RegistrationMonth</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Brand</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>DateCreated</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>NumberOfPictures</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>PostalCode</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>LastSeen</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



-- Приемлемый уровень нулевых значений. Можно будет заменить на модальное значение на этапе построения паплайна для модели

### Обработка типов данных


```python
# Преобразование и Удаление излишних данных о времени

df.DateCrawled = pd.to_datetime(df.DateCrawled).dt.date
df.DateCreated = pd.to_datetime(df.DateCreated).dt.date
df.LastSeen = pd.to_datetime(df.LastSeen).dt.date
```

### Анализ уникальных значений


```python
# Уникальные значения в колонках типа объект

temp = df.select_dtypes(include='object').columns
print('Уникальные значения в колонках типа объект:\n\n')

for name in temp:
    length_unique_values = len(df[name].unique())
    
    if length_unique_values < 20:
        print(f"{name}: {df[name].unique()}\n")
```

    Уникальные значения в колонках типа объект:
    
    
    VehicleType: [nan 'coupe' 'suv' 'small' 'sedan' 'convertible' 'bus' 'wagon' 'other']
    
    Gearbox: ['manual' 'auto' nan]
    
    FuelType: ['petrol' 'gasoline' nan 'lpg' 'other' 'hybrid' 'cng' 'electric']
    
    Repaired: [nan 'yes' 'no']
    
    

-- Нужно только заполнить пустые значения. 


```python
df = df.fillna('no_information') # Заполнение заглушкой пустых значений
```

-- "petrol" используется в британском английском, а "gasoline" — в американском английском. Можно их объединить


```python
# Замена значений
df.FuelType = df.FuelType.replace('petrol', 'gasoline')
```


```python
df.FuelType.unique()
```




    array(['gasoline', 'no_information', 'lpg', 'other', 'hybrid', 'cng',
           'electric'], dtype=object)



### Удаление дубликатов


```python
sum_deleted_row = 0
```


```python
# дубликаты
df[df.duplicated()] 
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
      <th>DateCrawled</th>
      <th>Price</th>
      <th>VehicleType</th>
      <th>RegistrationYear</th>
      <th>Gearbox</th>
      <th>Power</th>
      <th>Model</th>
      <th>Kilometer</th>
      <th>RegistrationMonth</th>
      <th>FuelType</th>
      <th>Brand</th>
      <th>Repaired</th>
      <th>DateCreated</th>
      <th>NumberOfPictures</th>
      <th>PostalCode</th>
      <th>LastSeen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8803</th>
      <td>2016-03-12</td>
      <td>1700</td>
      <td>sedan</td>
      <td>2004</td>
      <td>manual</td>
      <td>0</td>
      <td>mondeo</td>
      <td>150000</td>
      <td>9</td>
      <td>gasoline</td>
      <td>ford</td>
      <td>no_information</td>
      <td>2016-03-12</td>
      <td>0</td>
      <td>53520</td>
      <td>2016-03-12</td>
    </tr>
    <tr>
      <th>9413</th>
      <td>2016-03-21</td>
      <td>3399</td>
      <td>small</td>
      <td>2002</td>
      <td>manual</td>
      <td>90</td>
      <td>one</td>
      <td>150000</td>
      <td>8</td>
      <td>gasoline</td>
      <td>mini</td>
      <td>no</td>
      <td>2016-03-21</td>
      <td>0</td>
      <td>45739</td>
      <td>2016-03-23</td>
    </tr>
    <tr>
      <th>10699</th>
      <td>2016-03-22</td>
      <td>1800</td>
      <td>bus</td>
      <td>1997</td>
      <td>auto</td>
      <td>100</td>
      <td>vito</td>
      <td>150000</td>
      <td>8</td>
      <td>gasoline</td>
      <td>mercedes_benz</td>
      <td>no</td>
      <td>2016-03-22</td>
      <td>0</td>
      <td>22885</td>
      <td>2016-03-22</td>
    </tr>
    <tr>
      <th>10773</th>
      <td>2016-03-24</td>
      <td>16000</td>
      <td>sedan</td>
      <td>2014</td>
      <td>auto</td>
      <td>150</td>
      <td>a3</td>
      <td>20000</td>
      <td>3</td>
      <td>gasoline</td>
      <td>audi</td>
      <td>no</td>
      <td>2016-03-24</td>
      <td>0</td>
      <td>10115</td>
      <td>2016-03-24</td>
    </tr>
    <tr>
      <th>10912</th>
      <td>2016-03-26</td>
      <td>0</td>
      <td>coupe</td>
      <td>1991</td>
      <td>manual</td>
      <td>156</td>
      <td>other</td>
      <td>150000</td>
      <td>6</td>
      <td>gasoline</td>
      <td>toyota</td>
      <td>no_information</td>
      <td>2016-03-26</td>
      <td>0</td>
      <td>65929</td>
      <td>2016-04-07</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>354274</th>
      <td>2016-03-12</td>
      <td>7399</td>
      <td>wagon</td>
      <td>2006</td>
      <td>auto</td>
      <td>96</td>
      <td>a4</td>
      <td>125000</td>
      <td>12</td>
      <td>gasoline</td>
      <td>audi</td>
      <td>no</td>
      <td>2016-03-12</td>
      <td>0</td>
      <td>45699</td>
      <td>2016-03-29</td>
    </tr>
    <tr>
      <th>354281</th>
      <td>2016-03-07</td>
      <td>6500</td>
      <td>sedan</td>
      <td>1993</td>
      <td>auto</td>
      <td>286</td>
      <td>5er</td>
      <td>150000</td>
      <td>1</td>
      <td>gasoline</td>
      <td>bmw</td>
      <td>no</td>
      <td>2016-03-07</td>
      <td>0</td>
      <td>81249</td>
      <td>2016-04-05</td>
    </tr>
    <tr>
      <th>354300</th>
      <td>2016-03-26</td>
      <td>8940</td>
      <td>sedan</td>
      <td>2009</td>
      <td>manual</td>
      <td>80</td>
      <td>golf</td>
      <td>60000</td>
      <td>8</td>
      <td>gasoline</td>
      <td>volkswagen</td>
      <td>no</td>
      <td>2016-03-26</td>
      <td>0</td>
      <td>27299</td>
      <td>2016-04-06</td>
    </tr>
    <tr>
      <th>354317</th>
      <td>2016-03-19</td>
      <td>5990</td>
      <td>small</td>
      <td>2009</td>
      <td>manual</td>
      <td>86</td>
      <td>ibiza</td>
      <td>90000</td>
      <td>6</td>
      <td>gasoline</td>
      <td>seat</td>
      <td>no</td>
      <td>2016-03-19</td>
      <td>0</td>
      <td>51371</td>
      <td>2016-03-24</td>
    </tr>
    <tr>
      <th>354363</th>
      <td>2016-03-27</td>
      <td>1150</td>
      <td>bus</td>
      <td>2000</td>
      <td>manual</td>
      <td>0</td>
      <td>zafira</td>
      <td>150000</td>
      <td>3</td>
      <td>gasoline</td>
      <td>opel</td>
      <td>no</td>
      <td>2016-03-27</td>
      <td>0</td>
      <td>26624</td>
      <td>2016-03-29</td>
    </tr>
  </tbody>
</table>
<p>6832 rows × 16 columns</p>
</div>




```python
cnt_dup = sum(df.duplicated())
print(f"Кол-во дубликатов: {cnt_dup}")

alls = df.shape[0]
print(f"Кол-во записей: {alls}")
```

    Кол-во дубликатов: 6832
    Кол-во записей: 354369
    


```python
# Удаление явных дубликатов
df = df.drop_duplicates() 

sum_deleted_row += cnt_dup # сумма удалённых строк
```


```python
cnt_dup = sum(df.duplicated())
print(f"Кол-во дубликатов после удаления: {cnt_dup}")

alls = df.shape[0]
print(f"Кол-во записей после удаления: {alls}")
```

    Кол-во дубликатов после удаления: 0
    Кол-во записей после удаления: 347537
    

## Статистический анализ данных


```python
# Признаки для статистического анализа
df.head()
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
      <th>DateCrawled</th>
      <th>Price</th>
      <th>VehicleType</th>
      <th>RegistrationYear</th>
      <th>Gearbox</th>
      <th>Power</th>
      <th>Model</th>
      <th>Kilometer</th>
      <th>RegistrationMonth</th>
      <th>FuelType</th>
      <th>Brand</th>
      <th>Repaired</th>
      <th>DateCreated</th>
      <th>NumberOfPictures</th>
      <th>PostalCode</th>
      <th>LastSeen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-24</td>
      <td>480</td>
      <td>no_information</td>
      <td>1993</td>
      <td>manual</td>
      <td>0</td>
      <td>golf</td>
      <td>150000</td>
      <td>0</td>
      <td>gasoline</td>
      <td>volkswagen</td>
      <td>no_information</td>
      <td>2016-03-24</td>
      <td>0</td>
      <td>70435</td>
      <td>2016-04-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-24</td>
      <td>18300</td>
      <td>coupe</td>
      <td>2011</td>
      <td>manual</td>
      <td>190</td>
      <td>no_information</td>
      <td>125000</td>
      <td>5</td>
      <td>gasoline</td>
      <td>audi</td>
      <td>yes</td>
      <td>2016-03-24</td>
      <td>0</td>
      <td>66954</td>
      <td>2016-04-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-14</td>
      <td>9800</td>
      <td>suv</td>
      <td>2004</td>
      <td>auto</td>
      <td>163</td>
      <td>grand</td>
      <td>125000</td>
      <td>8</td>
      <td>gasoline</td>
      <td>jeep</td>
      <td>no_information</td>
      <td>2016-03-14</td>
      <td>0</td>
      <td>90480</td>
      <td>2016-04-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-17</td>
      <td>1500</td>
      <td>small</td>
      <td>2001</td>
      <td>manual</td>
      <td>75</td>
      <td>golf</td>
      <td>150000</td>
      <td>6</td>
      <td>gasoline</td>
      <td>volkswagen</td>
      <td>no</td>
      <td>2016-03-17</td>
      <td>0</td>
      <td>91074</td>
      <td>2016-03-17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-31</td>
      <td>3600</td>
      <td>small</td>
      <td>2008</td>
      <td>manual</td>
      <td>69</td>
      <td>fabia</td>
      <td>90000</td>
      <td>7</td>
      <td>gasoline</td>
      <td>skoda</td>
      <td>no</td>
      <td>2016-03-31</td>
      <td>0</td>
      <td>60437</td>
      <td>2016-04-06</td>
    </tr>
  </tbody>
</table>
</div>



### Цена (целевой признак) 


```python
plt.figure(figsize=(7, 4)) 
sns.histplot(data=df['Price'], bins=60)
plt.title(f'Гистограмма Price')
plt.xlabel('Цена (евро)')
plt.ylabel('Частота');
```


    
![png](output_31_0.png)
    


-- Чаще всего продаются машины с наиболее низкими ценами. Вероятно, естественное положение дел для рынка автомобилей.


-- Странным является только наличие большого кол-ва авто со слишком низкой ценой (ниже 50 евро), что не естественно для рынка.

Их стоит удалить чтобы они не мешали предсказывать реальную цену авто. 

В противном случае некоторые пользователи могут получить негативный опыт, при использовании приложения, от того что их машина стоит абсолютный 0.


```python
df = df.loc[df['Price']>=50]
```

### Тип кузова


```python
plt.figure(figsize=(7, 4)) 
sns.countplot(data=df, x='VehicleType', order=df['VehicleType'].value_counts().index)
plt.title(f'Частота значений VehicleType')
plt.ylabel('Кол-во значений')
plt.xlabel('Тип кузова')
plt.xticks(rotation=45);
```


    
![png](output_35_0.png)
    



```python
plt.figure(figsize=(7, 4)) 
sns.boxplot(data=df, x='VehicleType', y='Price', order=df['VehicleType'].value_counts().index)
plt.title('Распределение цен по типам кузова')
plt.xticks(rotation=45) 
plt.show()
```


    
![png](output_36_0.png)
    


### Год регистрации авто


```python
plt.figure(figsize=(7, 4)) 
sns.histplot(data=df, x='RegistrationYear', bins=100)
plt.title(f'Распределение значений RegistrationYear')
plt.xlabel('Год регистрации')
plt.ylabel('Кол-во значений');
```


    
![png](output_38_0.png)
    



```python
summ = df.loc[df['RegistrationYear']>2016, 'RegistrationYear'].shape[0] + \
    df.loc[df['RegistrationYear']<1970, 'RegistrationYear'].shape[0]

print('Кол-во некорректных строк:')
print(summ)
```

    Кол-во некорректных строк:
    14688
    


```python
# Удаление строк с ошибками в значениях
df = df.loc[df['RegistrationYear']>1970]
df = df.loc[df['RegistrationYear']<2017]

sum_deleted_row += summ # сумма удалённых строк
```


```python
plt.figure(figsize=(7, 4)) 
sns.histplot(data=df['RegistrationYear'], bins=50)
plt.title(f'Распределение значений RegistrationYear')
plt.xlabel('Год регистрации')
plt.ylabel('Кол-во значений');
```


    
![png](output_41_0.png)
    



```python
plt.figure(figsize=(7, 4)) 
plt.hexbin(df['RegistrationYear'], df['Price'], gridsize=40, cmap='Reds')
plt.colorbar(label='Кол-во значений')
plt.xlabel('Год регистрации')
plt.ylabel('Цена (евро)')
plt.title('Корреляция и плотность данных RegistrationYear и Price');
```


    
![png](output_42_0.png)
    


Здесь видна аномалия с данными после 2016 года. Важно все выше удалить.

### Тип коробки передач


```python
plt.figure(figsize=(7, 4)) 
sns.countplot(data=df, x='Gearbox')
plt.title(f'Частота значений Gearbox')
plt.ylabel('Частота значений');
```


    
![png](output_45_0.png)
    


### Мощность двигателя


```python
summ = df[df.Power > 500].shape[0] + df[df.Power < 30].shape[0]

print('Кол-во некорректных строк:')
print(summ)

# Удаление строк с ошибками в значениях
df = df[df.Power < 500]
df = df[df.Power > 30]

sum_deleted_row += summ # сумма удалённых строк
```

    Кол-во некорректных строк:
    32218
    

 Значения мощности двигателя более 500 л.с. и менее 30 л.с. считаются нереалистичными для двигателя обычных машин, поэтому они были удалены.


```python
plt.figure(figsize=(7, 4)) 
sns.histplot(data=df, x='Power', bins=50)
plt.title(f'Частота значений Power')
plt.ylabel('Частота значений');
```


    
![png](output_49_0.png)
    


### Модель 


```python
top_ten = df.Model.value_counts()[:10].index.tolist()
```


```python
plt.figure(figsize=(7, 4)) 
sns.countplot(data=df.loc[df['Model'].isin(top_ten)], x='Model', 
              order=df.loc[df['Model'].isin(top_ten)].Model.value_counts().index)
plt.title(f'Частота значений топ-10 Model')
plt.ylabel('Кол-во значений')
plt.xlabel('Модель')
plt.xticks(rotation=45);
```


    
![png](output_52_0.png)
    


### Пробег


```python
plt.figure(figsize=(7, 4)) 
sns.histplot(data=df, x='Kilometer', bins=50)
plt.title(f'Частота значений Kilometer')
plt.ylabel('Частота значений');
```


    
![png](output_54_0.png)
    



```python
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='Kilometer', y='Price')
plt.title(f'Распределение значений Kilometer к Price')
plt.xticks(rotation=45);
```


    
![png](output_55_0.png)
    


### Топливо


```python
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='FuelType', order=df.FuelType.value_counts().index)
plt.title(f'Частота значений FuelType')
plt.ylabel('Кол-во значений')
plt.xlabel('FuelType')
plt.xticks(rotation=45);
```


    
![png](output_57_0.png)
    



```python
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='FuelType', y='Price')
plt.title(f'Корреляция FuelType к Price')
plt.xticks(rotation=45);
```


    
![png](output_58_0.png)
    


### Бренд


```python
plt.figure(figsize=(10, 10))
sns.countplot(data=df, y='Brand', order=df.Brand.value_counts().index)
plt.title(f'Частота значений Brand')
plt.xlabel('Кол-во значений')
plt.ylabel('Бренд');
```


    
![png](output_60_0.png)
    



```python
plt.figure(figsize=(10, 10))
sns.boxplot(data=df, y='Brand', x='Price')
plt.title(f'Распределение значений Brand к Price');
```


    
![png](output_61_0.png)
    


### Была машина в ремонте или нет


```python
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='Repaired', order=df.Repaired.value_counts().index)
plt.title(f'Частота значений Repaired')
plt.ylabel('Кол-во значений')
plt.xlabel('Repaired');
```


    
![png](output_63_0.png)
    



```python
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='Repaired', y='Price')
plt.title(f'Корреляция Repaired к Price');
```


    
![png](output_64_0.png)
    


## Корреляционный анализ данных


```python
col_to_drop = ['NumberOfPictures', 'PostalCode', 'DateCrawled', 'DateCreated', 'LastSeen', 'RegistrationMonth']

res = phik_matrix(df.drop(col_to_drop, axis=1), interval_cols=['Price', 'Power', 'Kilometer'])
```


```python
plt.figure(figsize=(8, 5))

sns.heatmap(res, annot=True, cmap='OrRd');
```


    
![png](output_67_0.png)
    


-- Наиболее коррелирующие признаки:

-- RegistrationYear -- 0.66

-- Power -- 0.51

-- Model -- 0.58

Чтобы избежать ухудшения обучаемости вследствии мультиколлинеарности признаков -- стоит убрать признак "Brand" (в частности изза его меньшей информативности по сравнению с "Model" - моделью машины)


```python
df_cleared = df.drop(col_to_drop+['Brand'], axis=1)
```


```python
df.head()
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
      <th>DateCrawled</th>
      <th>Price</th>
      <th>VehicleType</th>
      <th>RegistrationYear</th>
      <th>Gearbox</th>
      <th>Power</th>
      <th>Model</th>
      <th>Kilometer</th>
      <th>RegistrationMonth</th>
      <th>FuelType</th>
      <th>Brand</th>
      <th>Repaired</th>
      <th>DateCreated</th>
      <th>NumberOfPictures</th>
      <th>PostalCode</th>
      <th>LastSeen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2016-03-24</td>
      <td>18300</td>
      <td>coupe</td>
      <td>2011</td>
      <td>manual</td>
      <td>190</td>
      <td>no_information</td>
      <td>125000</td>
      <td>5</td>
      <td>gasoline</td>
      <td>audi</td>
      <td>yes</td>
      <td>2016-03-24</td>
      <td>0</td>
      <td>66954</td>
      <td>2016-04-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-14</td>
      <td>9800</td>
      <td>suv</td>
      <td>2004</td>
      <td>auto</td>
      <td>163</td>
      <td>grand</td>
      <td>125000</td>
      <td>8</td>
      <td>gasoline</td>
      <td>jeep</td>
      <td>no_information</td>
      <td>2016-03-14</td>
      <td>0</td>
      <td>90480</td>
      <td>2016-04-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-17</td>
      <td>1500</td>
      <td>small</td>
      <td>2001</td>
      <td>manual</td>
      <td>75</td>
      <td>golf</td>
      <td>150000</td>
      <td>6</td>
      <td>gasoline</td>
      <td>volkswagen</td>
      <td>no</td>
      <td>2016-03-17</td>
      <td>0</td>
      <td>91074</td>
      <td>2016-03-17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-31</td>
      <td>3600</td>
      <td>small</td>
      <td>2008</td>
      <td>manual</td>
      <td>69</td>
      <td>fabia</td>
      <td>90000</td>
      <td>7</td>
      <td>gasoline</td>
      <td>skoda</td>
      <td>no</td>
      <td>2016-03-31</td>
      <td>0</td>
      <td>60437</td>
      <td>2016-04-06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-04-04</td>
      <td>650</td>
      <td>sedan</td>
      <td>1995</td>
      <td>manual</td>
      <td>102</td>
      <td>3er</td>
      <td>150000</td>
      <td>10</td>
      <td>gasoline</td>
      <td>bmw</td>
      <td>yes</td>
      <td>2016-04-04</td>
      <td>0</td>
      <td>33775</td>
      <td>2016-04-06</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cleared.head()
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
      <th>Price</th>
      <th>VehicleType</th>
      <th>RegistrationYear</th>
      <th>Gearbox</th>
      <th>Power</th>
      <th>Model</th>
      <th>Kilometer</th>
      <th>FuelType</th>
      <th>Repaired</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>18300</td>
      <td>coupe</td>
      <td>2011</td>
      <td>manual</td>
      <td>190</td>
      <td>no_information</td>
      <td>125000</td>
      <td>gasoline</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9800</td>
      <td>suv</td>
      <td>2004</td>
      <td>auto</td>
      <td>163</td>
      <td>grand</td>
      <td>125000</td>
      <td>gasoline</td>
      <td>no_information</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1500</td>
      <td>small</td>
      <td>2001</td>
      <td>manual</td>
      <td>75</td>
      <td>golf</td>
      <td>150000</td>
      <td>gasoline</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3600</td>
      <td>small</td>
      <td>2008</td>
      <td>manual</td>
      <td>69</td>
      <td>fabia</td>
      <td>90000</td>
      <td>gasoline</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>650</td>
      <td>sedan</td>
      <td>1995</td>
      <td>manual</td>
      <td>102</td>
      <td>3er</td>
      <td>150000</td>
      <td>gasoline</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cleared.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 288081 entries, 1 to 354368
    Data columns (total 9 columns):
     #   Column            Non-Null Count   Dtype 
    ---  ------            --------------   ----- 
     0   Price             288081 non-null  int64 
     1   VehicleType       288081 non-null  object
     2   RegistrationYear  288081 non-null  int64 
     3   Gearbox           288081 non-null  object
     4   Power             288081 non-null  int64 
     5   Model             288081 non-null  object
     6   Kilometer         288081 non-null  int64 
     7   FuelType          288081 non-null  object
     8   Repaired          288081 non-null  object
    dtypes: int64(4), object(5)
    memory usage: 22.0+ MB
    

## Подготовка данных


```python
X = df_cleared.drop(['Price'], axis=1)
y = df_cleared.Price
```


```python
X
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
      <th>VehicleType</th>
      <th>RegistrationYear</th>
      <th>Gearbox</th>
      <th>Power</th>
      <th>Model</th>
      <th>Kilometer</th>
      <th>FuelType</th>
      <th>Repaired</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>coupe</td>
      <td>2011</td>
      <td>manual</td>
      <td>190</td>
      <td>no_information</td>
      <td>125000</td>
      <td>gasoline</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>suv</td>
      <td>2004</td>
      <td>auto</td>
      <td>163</td>
      <td>grand</td>
      <td>125000</td>
      <td>gasoline</td>
      <td>no_information</td>
    </tr>
    <tr>
      <th>3</th>
      <td>small</td>
      <td>2001</td>
      <td>manual</td>
      <td>75</td>
      <td>golf</td>
      <td>150000</td>
      <td>gasoline</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>small</td>
      <td>2008</td>
      <td>manual</td>
      <td>69</td>
      <td>fabia</td>
      <td>90000</td>
      <td>gasoline</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sedan</td>
      <td>1995</td>
      <td>manual</td>
      <td>102</td>
      <td>3er</td>
      <td>150000</td>
      <td>gasoline</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>354361</th>
      <td>no_information</td>
      <td>2016</td>
      <td>auto</td>
      <td>150</td>
      <td>159</td>
      <td>150000</td>
      <td>no_information</td>
      <td>no</td>
    </tr>
    <tr>
      <th>354362</th>
      <td>sedan</td>
      <td>2004</td>
      <td>manual</td>
      <td>225</td>
      <td>leon</td>
      <td>150000</td>
      <td>gasoline</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>354366</th>
      <td>convertible</td>
      <td>2000</td>
      <td>auto</td>
      <td>101</td>
      <td>fortwo</td>
      <td>125000</td>
      <td>gasoline</td>
      <td>no</td>
    </tr>
    <tr>
      <th>354367</th>
      <td>bus</td>
      <td>1996</td>
      <td>manual</td>
      <td>102</td>
      <td>transporter</td>
      <td>150000</td>
      <td>gasoline</td>
      <td>no</td>
    </tr>
    <tr>
      <th>354368</th>
      <td>wagon</td>
      <td>2002</td>
      <td>manual</td>
      <td>100</td>
      <td>golf</td>
      <td>150000</td>
      <td>gasoline</td>
      <td>no_information</td>
    </tr>
  </tbody>
</table>
<p>288081 rows × 8 columns</p>
</div>




```python
ohe_cols = ['Gearbox','Repaired']
num_cols = ['Power', 'Kilometer', 'RegistrationYear']
targ_cols = ['VehicleType', 'Model', 'FuelType']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)


num_pipe = Pipeline(steps=[
    ('simpleImputer_num', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('num', StandardScaler())
])

targ_pipe = Pipeline(steps=[
    ('simpleImputer_targ', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('target', TargetEncoder())
])

ohe_pipe = Pipeline(steps=[
    ('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first'))
])



data_preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, num_cols),
        ('ohe', ohe_pipe, ohe_cols),
        ('target', targ_pipe, targ_cols)
    ]
)


pipe_final = Pipeline(steps=[
    ('preprocessor', data_preprocessor),
    ('models', LinearRegression())
])
```


```python
param_grid = [
    {
        'models': [CatBoostRegressor(random_state=42, verbose=1000)],
        'models__iterations': [100, 200],
        'models__depth': [6, 8],
        'models__learning_rate': [0.1, 0.2],
        'preprocessor__num': [StandardScaler(), MinMaxScaler()]
    },
    {
        'models': [LGBMRegressor(random_state=42, verbose=1, force_row_wise=True)],
        'models__num_leaves': [31, 63],
        'models__learning_rate': [0.1, 0.2],
        'models__n_estimators': [100, 200],
        'preprocessor__num': [StandardScaler(), MinMaxScaler()]
    },
    {
        'models': [LinearRegression()],
        'preprocessor__num': [StandardScaler(), MinMaxScaler()]
    }
]
```

## Обучение моделей


```python
grid_search = GridSearchCV(
    pipe_final,
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
```


```python
grid_search.fit(X_train, y_train)
```

    [LightGBM] [Info] Total Bins 566
    [LightGBM] [Info] Number of data points in the train set: 216060, number of used features: 10
    [LightGBM] [Info] Start training from score 4841.207137
    




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleImputer_num&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                         (&#x27;num&#x27;,
                                                                                          StandardScaler())]),
                                                                         [&#x27;Power&#x27;,
                                                                          &#x27;Kilometer&#x27;,
                                                                          &#x27;RegistrationYear&#x27;]),
                                                                        (&#x27;ohe&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                         (&#x27;ohe&#x27;,
                                                                                          OneHotEncode...
                                                MinMaxScaler()]},
                         {&#x27;models&#x27;: [LGBMRegressor(force_row_wise=True,
                                                   random_state=42,
                                                   verbose=1)],
                          &#x27;models__learning_rate&#x27;: [0.1, 0.2],
                          &#x27;models__n_estimators&#x27;: [100, 200],
                          &#x27;models__num_leaves&#x27;: [31, 63],
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler()]},
                         {&#x27;models&#x27;: [LinearRegression()],
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler()]}],
             scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleImputer_num&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                         (&#x27;num&#x27;,
                                                                                          StandardScaler())]),
                                                                         [&#x27;Power&#x27;,
                                                                          &#x27;Kilometer&#x27;,
                                                                          &#x27;RegistrationYear&#x27;]),
                                                                        (&#x27;ohe&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                                         (&#x27;ohe&#x27;,
                                                                                          OneHotEncode...
                                                MinMaxScaler()]},
                         {&#x27;models&#x27;: [LGBMRegressor(force_row_wise=True,
                                                   random_state=42,
                                                   verbose=1)],
                          &#x27;models__learning_rate&#x27;: [0.1, 0.2],
                          &#x27;models__n_estimators&#x27;: [100, 200],
                          &#x27;models__num_leaves&#x27;: [31, 63],
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler()]},
                         {&#x27;models&#x27;: [LinearRegression()],
                          &#x27;preprocessor__num&#x27;: [StandardScaler(),
                                                MinMaxScaler()]}],
             scoring=&#x27;neg_root_mean_squared_error&#x27;)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                                  [&#x27;Power&#x27;, &#x27;Kilometer&#x27;,
                                                   &#x27;RegistrationYear&#x27;]),
                                                 (&#x27;ohe&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;ohe&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;))]),
                                                  [&#x27;Gearbox&#x27;, &#x27;Repaired&#x27;]),
                                                 (&#x27;target&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_targ&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;target&#x27;,
                                                                   TargetEncoder())]),
                                                  [&#x27;VehicleType&#x27;, &#x27;Model&#x27;,
                                                   &#x27;FuelType&#x27;])])),
                (&#x27;models&#x27;,
                 LGBMRegressor(force_row_wise=True, learning_rate=0.2,
                               n_estimators=200, num_leaves=63, random_state=42,
                               verbose=1))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                 [&#x27;Power&#x27;, &#x27;Kilometer&#x27;, &#x27;RegistrationYear&#x27;]),
                                (&#x27;ohe&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;ohe&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;))]),
                                 [&#x27;Gearbox&#x27;, &#x27;Repaired&#x27;]),
                                (&#x27;target&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_targ&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;target&#x27;, TargetEncoder())]),
                                 [&#x27;VehicleType&#x27;, &#x27;Model&#x27;, &#x27;FuelType&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>num</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Power&#x27;, &#x27;Kilometer&#x27;, &#x27;RegistrationYear&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>ohe</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Gearbox&#x27;, &#x27;Repaired&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;first&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>target</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;VehicleType&#x27;, &#x27;Model&#x27;, &#x27;FuelType&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>TargetEncoder</div></div></label><div class="sk-toggleable__content fitted"><pre>TargetEncoder()</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LGBMRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>LGBMRegressor(force_row_wise=True, learning_rate=0.2, n_estimators=200,
              num_leaves=63, random_state=42, verbose=1)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
best_model = grid_search.best_estimator_
```


```python
# Обучаем на тренировочных данных и трансформируем тестовые признаки 
data_preprocessor.fit(X_train, y_train)  
X_test_transformed = data_preprocessor.transform(X_test)

# Сборка трансформированных призаков перед использованием 
X_test_transformed = pd.DataFrame(X_test_transformed, columns=data_preprocessor.get_feature_names_out())
```


```python
X_test_transformed.columns # Проверка
```




    Index(['num__Power', 'num__Kilometer', 'num__RegistrationYear',
           'ohe__Gearbox_manual', 'ohe__Gearbox_no_information',
           'ohe__Repaired_no_information', 'ohe__Repaired_yes', 'target__0',
           'target__1', 'target__2'],
          dtype='object')




```python
# Использование лучшей модели вручную
y_pred = grid_search.best_estimator_.named_steps['models'].predict(X_test_transformed)
y_error = y_test - y_pred # Расчёт ошибки прогноза
```


```python
print(f"Средняя Метрика RMSE на тренировочных данных: {round(grid_search.best_score_, 1)*-1}")
```

    Средняя Метрика RMSE на тренировочных данных: 1558.5
    


```python
rmse_metric = mean_squared_error(y_test, y_pred) ** 0.5

print(f'Метрика RMSE на тестовых данных: {round(rmse_metric, 1)}')
```

    Метрика RMSE на тестовых данных: 1571.6
    


```python
plt.figure(figsize=(7, 4))
sns.histplot(y_error, bins=100)
plt.title('Частота значений ошибки')
plt.xlabel('Значение Ошибки')
plt.ylabel('Кол-во');
```


    
![png](output_88_0.png)
    



```python
result = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred,
    'y_error': y_error
})
```


```python
result
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
      <th>y_test</th>
      <th>y_pred</th>
      <th>y_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>303370</th>
      <td>3590</td>
      <td>3975.046435</td>
      <td>-385.046435</td>
    </tr>
    <tr>
      <th>331363</th>
      <td>1999</td>
      <td>1925.494206</td>
      <td>73.505794</td>
    </tr>
    <tr>
      <th>351022</th>
      <td>4355</td>
      <td>8225.317912</td>
      <td>-3870.317912</td>
    </tr>
    <tr>
      <th>2223</th>
      <td>2290</td>
      <td>2149.155245</td>
      <td>140.844755</td>
    </tr>
    <tr>
      <th>221932</th>
      <td>4999</td>
      <td>4416.550899</td>
      <td>582.449101</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23938</th>
      <td>1275</td>
      <td>562.781482</td>
      <td>712.218518</td>
    </tr>
    <tr>
      <th>174462</th>
      <td>2200</td>
      <td>2578.771755</td>
      <td>-378.771755</td>
    </tr>
    <tr>
      <th>250110</th>
      <td>650</td>
      <td>490.443974</td>
      <td>159.556026</td>
    </tr>
    <tr>
      <th>203129</th>
      <td>5000</td>
      <td>6078.379010</td>
      <td>-1078.379010</td>
    </tr>
    <tr>
      <th>65972</th>
      <td>2100</td>
      <td>3107.647906</td>
      <td>-1007.647906</td>
    </tr>
  </tbody>
</table>
<p>72021 rows × 3 columns</p>
</div>




```python
plt.figure(figsize=(8, 5))
ax = sns.lineplot(data=result[['y_test', 'y_pred']])
plt.title('График отклонения пресказанного значения')
ax.set_xlim(100, 400);
```


    
![png](output_91_0.png)
    



```python
res = pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
res[['rank_test_score', 'mean_fit_time', 'params', 'std_test_score']].head()
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
      <th>rank_test_score</th>
      <th>mean_fit_time</th>
      <th>params</th>
      <th>std_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>12.406437</td>
      <td>{'models': LGBMRegressor(force_row_wise=True, ...</td>
      <td>15.078209</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>9.195364</td>
      <td>{'models': LGBMRegressor(force_row_wise=True, ...</td>
      <td>15.171651</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>13.799182</td>
      <td>{'models': LGBMRegressor(force_row_wise=True, ...</td>
      <td>14.674502</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>14.154460</td>
      <td>{'models': LGBMRegressor(force_row_wise=True, ...</td>
      <td>15.906234</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>8.315635</td>
      <td>{'models': LGBMRegressor(force_row_wise=True, ...</td>
      <td>15.543761</td>
    </tr>
  </tbody>
</table>
</div>



#### Анализ важности признаков


```python
# оценка важности признаков через permutation_importance
result = permutation_importance(grid_search.best_estimator_.named_steps['models'], X_test_transformed, y_test, n_repeats=10, random_state=42)
importance = result.importances_mean
feature_names = X_test_transformed.columns


per_importance_df = pd.DataFrame(data={'Признак': feature_names,
                                       'Важность': np.around(importance, decimals=3)})

# важность признаков
per_importance_df.sort_values(by='Важность', ascending=False)
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
      <th>Признак</th>
      <th>Важность</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>num__RegistrationYear</td>
      <td>0.712</td>
    </tr>
    <tr>
      <th>0</th>
      <td>num__Power</td>
      <td>0.312</td>
    </tr>
    <tr>
      <th>8</th>
      <td>target__1</td>
      <td>0.145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>num__Kilometer</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>7</th>
      <td>target__0</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ohe__Repaired_yes</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ohe__Gearbox_manual</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ohe__Repaired_no_information</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ohe__Gearbox_no_information</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>target__2</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>



### Цель

Приложение должно предлагать точные и быстрые оценки, чтобы повысить доверие клиентов к сервису и привлечь новых клиентов.

### Результаты

- **CatBoost** показал отличное качество предсказания с RMSE, что ниже порогового значения 2500. Лучшие показатели относительно времени обучения, времени прогнозирования и ошибки модели.


- **LightGBM** продемонстрировал высочайшее качество предсказания с RMSE. Однако время обучения было самым длинным среди всех моделей, а время прогнозирования — чуть дольше, чем у CatBoost.


- **Линейная регрессия** обучалась и предсказывала очень быстро, но её точность оставляет желать лучшего, так как RMSE значительно превышает порог в 2500.

### На основании полученных результатов можно сделать следующие выводы:

- **LightGBM и CatBoost** предоставляют отличное качество предсказаний, соответствующее требованиям заказчика.


- **LightGBM** имеет преимущество по качеству обучения и прогнозирования, что делает его предпочтительным выбором среди моделей градиентного бустинга.


- **Линейная регрессия** не удовлетворяет требованиям по точности, хотя и является самой быстрой моделью по времени обучения и прогнозирования.

### Лучшая модель

Модель **LightGBM** была разработана и обучена для решения задачи оценки стоимости авто. На основе тщательно проведенного анализа и тестирования, модель продемонстрировала высокие показатели точности и надежности, что делает её готовой к внедрению в сервис.

**LightGBM** лучше всего подходит по быстродействию и точности для имплементации в сервис по продаже автомобилей.



```python
best_model
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                                  [&#x27;Power&#x27;, &#x27;Kilometer&#x27;,
                                                   &#x27;RegistrationYear&#x27;]),
                                                 (&#x27;ohe&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;ohe&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;))]),
                                                  [&#x27;Gearbox&#x27;, &#x27;Repaired&#x27;]),
                                                 (&#x27;target&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_targ&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;target&#x27;,
                                                                   TargetEncoder())]),
                                                  [&#x27;VehicleType&#x27;, &#x27;Model&#x27;,
                                                   &#x27;FuelType&#x27;])])),
                (&#x27;models&#x27;,
                 LGBMRegressor(force_row_wise=True, learning_rate=0.2,
                               n_estimators=200, num_leaves=63, random_state=42,
                               verbose=1))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                                  [&#x27;Power&#x27;, &#x27;Kilometer&#x27;,
                                                   &#x27;RegistrationYear&#x27;]),
                                                 (&#x27;ohe&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;ohe&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;))]),
                                                  [&#x27;Gearbox&#x27;, &#x27;Repaired&#x27;]),
                                                 (&#x27;target&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleImputer_targ&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;target&#x27;,
                                                                   TargetEncoder())]),
                                                  [&#x27;VehicleType&#x27;, &#x27;Model&#x27;,
                                                   &#x27;FuelType&#x27;])])),
                (&#x27;models&#x27;,
                 LGBMRegressor(force_row_wise=True, learning_rate=0.2,
                               n_estimators=200, num_leaves=63, random_state=42,
                               verbose=1))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;, StandardScaler(),
                                 [&#x27;Power&#x27;, &#x27;Kilometer&#x27;, &#x27;RegistrationYear&#x27;]),
                                (&#x27;ohe&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_ohe&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;ohe&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;))]),
                                 [&#x27;Gearbox&#x27;, &#x27;Repaired&#x27;]),
                                (&#x27;target&#x27;,
                                 Pipeline(steps=[(&#x27;simpleImputer_targ&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;target&#x27;, TargetEncoder())]),
                                 [&#x27;VehicleType&#x27;, &#x27;Model&#x27;, &#x27;FuelType&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>num</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Power&#x27;, &#x27;Kilometer&#x27;, &#x27;RegistrationYear&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>ohe</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Gearbox&#x27;, &#x27;Repaired&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;first&#x27;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>target</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;VehicleType&#x27;, &#x27;Model&#x27;, &#x27;FuelType&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>TargetEncoder</div></div></label><div class="sk-toggleable__content fitted"><pre>TargetEncoder()</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LGBMRegressor</div></div></label><div class="sk-toggleable__content fitted"><pre>LGBMRegressor(force_row_wise=True, learning_rate=0.2, n_estimators=200,
              num_leaves=63, random_state=42, verbose=1)</pre></div> </div></div></div></div></div></div>




```python
grid_search.best_params_
```




    {'models': LGBMRegressor(force_row_wise=True, random_state=42, verbose=1),
     'models__learning_rate': 0.2,
     'models__n_estimators': 200,
     'models__num_leaves': 63,
     'preprocessor__num': StandardScaler()}




```python

```
