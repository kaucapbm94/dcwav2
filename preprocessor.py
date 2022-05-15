from scipy.special import inv_boxcox1p
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy.stats import skew
from transliterate import translit
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from visualizer import Visualizer


class Preprocessor:
  def __init__(self, visualizer: Visualizer) -> None:
    self.visualizer = visualizer
    self.null_cell_drops_row_columns = [
        'max_floor',
        'floor',
        'build_year',
        'lat',
        'long',
        'district',
        # 'ceiling_height',
    ]
    self.cols_to_translit = [
        'residential_complex',
        'condition',
        'floor_type',
        'door',
        'building_type',
        'district',
    ]
    self.seed_columns = [
        'text',  # => replace with '' and compute text_len
        'intersection',  # convert to longitude / latitude
        # 'district',
        'street',
        'intersection',
        'house_number',
        'city',
        'address',
    ]
    self.fill_category_columns = [
        'residential_complex',
        'door',
        'floor_type',
        'condition',
    ]  # replace with true false
    self.to_drop_columns = [
        'living_area',  # too large missing value fraction => not significant
        'kitchen_area',  # too large missing value fraction => not significant
        'address',  # don't need it anymore
        'ceiling_height'
    ]
    self.allowed_b_types = ['монолитный', 'кирпичный', 'панельный', 'иное']

  def missing_values(self, df: DataFrame) -> DataFrame:
    # missing data
    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    exclude = self.null_cell_drops_row_columns + self.seed_columns + self.fill_category_columns
    labels_to_drop = missing_data[~missing_data.index.isin(exclude)][missing_data['Total'] > 1].index
    df = df.drop(columns=labels_to_drop)
    print(f"Dropped: {labels_to_drop} columns because of nulls\n")

    self.visualizer.plot_missing_data(missing_data)

    for ncdrc in self.null_cell_drops_row_columns:
      df = df.drop(df.loc[df[ncdrc].isnull()].index)
    print(f"Dropped rows with null on {self.null_cell_drops_row_columns} columns instead of dropping whole column\n")
    return df

  def transform_seeds(self, df: DataFrame) -> DataFrame:
    # replace text with length of text
    df['text'].fillna('', inplace=True)
    df['text_len'] = df.apply(lambda row: len(row['text']), axis=1)
    print(f"Added text_len\nDropped text, city columns\n")

    df.drop(columns=self.seed_columns, inplace=True)
    print(f"Dropped seed columns {self.seed_columns}")

    category_columns = []
    for category_column in category_columns:
      lbl = LabelEncoder()
      lbl.fit(list(df[category_column].values))
      df[category_column] = lbl.transform(list(df[category_column].values))
    print(f"Encoded category columns({category_columns})")
    return df

  def clean(self, df: DataFrame) -> DataFrame:
    # remove false scraped price
    price_upper_bound = 100000
    df = df.drop(index=df.loc[df['price'] <= price_upper_bound].index)
    print(f"Dropped rows with price <= {price_upper_bound}\n")

    # remove false scraped ceiling height, but not here
    ceiling_height_upperbound = 10
    ceiling_height_lowerbound = 0.5
    df2 = df.drop(index=df.loc[df['ceiling_height'] >= ceiling_height_upperbound].index)
    df2 = df2.drop(index=df2.loc[df2['ceiling_height'] <= ceiling_height_lowerbound].index)
    self.visualizer.scatter(df2, 'ceiling_height')
    print(f"df2: Dropped rows not in span: {ceiling_height_upperbound} <= ceiling_height <= {ceiling_height_lowerbound} m\n")

    # remove false scraped building type
    df = df[df.building_type.isin(self.allowed_b_types)]

    return df

  def set_category_None(self, df: DataFrame) -> DataFrame:
    for fill_category_column in self.fill_category_columns:
      df[fill_category_column].fillna("None", inplace=True)
      print(f"Replaced None values of \'{fill_category_column}\' category_column with string \'None\'\n")
    return df

  def resolve_skewness(self, df: DataFrame, rsk: bool) -> DataFrame:
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print(f"\nSkew in numerical features:\n{skewness.head(10)}\n")

    skewness = skewness[abs(skewness) > 0.75]
    skewed_features = skewness.index
    lam = 0.15
    if rsk:
      for skewed_feature in skewed_features:
        #all_data[feat] += 1
        df[skewed_feature] = boxcox1p(df[skewed_feature], lam)
      print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    # transform
    df['price_log1p'] = np.log1p(df.price)
    print(f'transformed price to price_log1p')

    return df

  def transliterate(self, df: DataFrame) -> DataFrame:
    # transliterate
    for col in self.cols_to_translit:
      df[col] = df[col].apply(lambda x: translit(x, 'ru', reversed=True))
    print(f"Translitted column names ({self.cols_to_translit}) to latin alphabet")
    return df

  def get_dummies(self, df: DataFrame) -> DataFrame:
    df.reset_index(drop=True, inplace=True)
    df = pd.get_dummies(df)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    return df

  def reverse(self, serie: Series, predict_col: str, unskewed: bool) -> Series:
    if predict_col == 'price_log1p':
      serie = np.expm1(serie)
    if unskewed:
      lam = 0.15
      serie = inv_boxcox1p(serie, lam)
    return serie
