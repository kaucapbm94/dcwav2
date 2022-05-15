from io import TextIOWrapper
from visualizer import Visualizer
from preprocessor import Preprocessor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pandas import DataFrame
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
import lightgbm as lgb
import warnings
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')


def ignore_warn(*args, **kwargs):
  pass


warnings.warn = ignore_warn
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
  def __init__(self, models):
    self.models = models

  # we define clones of the original models to fit the data in
  def fit(self, X, y):
    self.models_ = [clone(x) for x in self.models]

    # Train cloned base models
    for model in self.models_:
      model.fit(X, y)

    return self

  # Now we do the predictions for cloned models and average them
  def predict(self, X):
    predictions = np.column_stack([
        model.predict(X) for model in self.models_
    ])
    return np.mean(predictions, axis=1)


class NN:
  def __init__(self, predict_col: str, resolve_skewness: bool = True) -> None:
    self.df: DataFrame
    self.train_df: DataFrame
    self.predict_col = predict_col
    self.test_df: DataFrame
    self.x_train: DataFrame
    self.y_train: np.ndarray
    self.x_test: DataFrame
    self.visualizer = Visualizer()
    self.preprocessor = Preprocessor(self.visualizer)
    self.y_test: np.ndarray
    self.n_folds = 5
    self.final_drop_cols = ['price', 'price_log1p']
    self.rsk = resolve_skewness

    self.models = {
        'KernelRidge': KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
        'ElasticNet': make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)),
        'Lasso': make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1)),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5, verbose=-1),
        'XGBRegressor': xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1, verbose=-1),
        'LGBMRegressor': lgb.LGBMRegressor(objective='regression', num_leaves=5,  learning_rate=0.05, n_estimators=720,  max_bin=55, bagging_fraction=0.8,  bagging_freq=5, feature_fraction=0.2319,  feature_fraction_seed=9, bagging_seed=9,  min_data_in_leaf=6, min_sum_hessian_in_leaf=11),
    }
    self.averaged_model_names = ['ElasticNet', 'GradientBoostingRegressor', 'KernelRidge', 'Lasso']
    self.models['Averaged model'] = self.averaged_models = AveragingModels(models=(tuple([self.models[model_name] for model_name in self.averaged_model_names])))
    self.rmsle_cv_scores = {}
    self.r2_scores = {}

  def load(self) -> None:
    self.df = pd.read_csv('df3.csv')
    self.df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)

  def preprocess(self):
    df = self.df
    print(f"Df shape before preprocessing is : {df.shape}")
    print(f"df columns: {list(df.columns)}")

    print(self.df.describe())
    print(self.df.columns)

    self.visualizer.corr(self.df)

    self.visualizer.describe_column(self.df, 'price')
    self.visualizer.scatters(self.df)
    self.visualizer.std(self.df)

    df = self.preprocessor.clean(df)
    df = self.preprocessor.set_category_None(df)
    scatter__s = [
        'general_area',
        'internet',
        'floor',
        'room_count',
        # 'ceiling_height',
        'non_angular',
        'max_floor',
    ]
    for s in scatter__s:
      self.visualizer.scatter(self.df, s)

    df = self.preprocessor.missing_values(df)

    df = self.preprocessor.transform_seeds(df)
    df = self.preprocessor.resolve_skewness(df, self.rsk)

    df = self.preprocessor.transliterate(df)
    df = self.preprocessor.get_dummies(df)
    print(f"\n\nFinal columns: {list(df.columns)}\n\n")
    print(f"Df shape after preprocessing is : {df.shape}")
    self.df = df

  def split(self) -> None:
    self.train_df, self.test_df = train_test_split(self.df, test_size=0.2)
    self.x_train: DataFrame = self.train_df.drop(columns=self.final_drop_cols)
    self.y_train: np.ndarray = self.train_df[self.predict_col]
    self.x_test: DataFrame = self.test_df.drop(columns=self.final_drop_cols)
    self.y_test: np.ndarray = self.test_df[self.predict_col]

    print(f"Train size is : {self.train_df.shape}")
    print(f"Test size is : {self.test_df.shape}")

  def describe_post_proc(self) -> None:
    print(self.df.describe())
    self.visualizer.describe_column(self.df, 'price')
    self.visualizer.scatters(self.df)
    self.visualizer.scatter(self.df, 'general_area')
    # self.visualizer.scatter(self.df, 'internet')
    self.visualizer.scatter(self.df, 'floor')
    self.visualizer.scatter(self.df, 'room_count')
    # self.visualizer.scatter(self.df, 'ceiling_height')
    self.visualizer.std(self.df)
    print([column_name for column_name in list(self.df.columns) if column_name.startswith('district')])

  # Validation function

  def rmsle_cv(self, model):
    kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(self.train_df.values)
    # root mean squared error
    rmse = np.sqrt(-cross_val_score(model, self.train_df.values, self.y_train, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

  def show_scores(self):
    fout = open('results.txt', 'w')
    # print("Averaged base models score: {:.4f} ({:.4f})\n".format(self.score.mean(), self.score.std()))
    s = 'R square Accuracy | Mean Absolute Error Accuracy | Mean Squared Error Accuracy'
    print(s)
    # f.write(s)
    for model_name, score in self.r2_scores.items():
      self.compare_y_pred_y_test(score['y_pred'], model_name, fout)
      # print('\n' + '-'*10+model_name+'-'*10)
      # print('R square Accuracy: ', score['r2_score'])
      # print('Mean Absolute Error Accuracy: ', score['mean_absolute_error'])
      # print('Mean Squared Error Accuracy: ', score['mean_squared_error'], '\n')
    for model_name, score in self.rmsle_cv_scores.items():
      s = "{} score | mean | std\n".format(model_name)
      print(s)
      # f.write(s)
      s = "{:.4f}\t{:.4f}\n".format(score.mean(), score.std())
      print(s)
      fout.write(s)
    fout.close()
    # print scores without errors again
    # for model_name, score in nn.r2_scores.items():

  def rmsle_cv_score(self, model_name: str) -> None:
    model = self.models[model_name]
    score = self.rmsle_cv(model)
    print("{} score: mean={:.4f} std=({:.4f})\n".format(model_name, score.mean(), score.std()))
    self.rmsle_cv_scores[model_name] = score

  def compare_y_pred_y_test(self, y_pred: DataFrame, model_name: str, fout: TextIOWrapper = None) -> dict[str, float]:
    y_test = self.y_test.reset_index(drop=True)
    y_pred = pd.DataFrame(y_pred, columns=['Predict'])

    score = {
        'r2_score': r2_score(y_test, y_pred),
        'mean_absolute_error': mean_absolute_error(y_test, y_pred),
        'mean_squared_error': mean_squared_error(y_test, y_pred),
        'y_pred': y_pred,
    }
    s = '\n' + '-'*10+model_name+'-'*10 + '\n'
    print(s)
    # f.write(s)

    s = f"{round(score['r2_score'], 2)}\t{round(score['mean_absolute_error'], 4)}\t{round(score['mean_squared_error'], 6)}\n"
    print(s)
    if fout:
      fout.write(s)

    y_test_y_head = pd.concat([self.preprocessor.reverse(y_test, self.predict_col, True), self.preprocessor.reverse(y_pred, self.predict_col, True)], axis=1)

    print(y_test_y_head.head(5))
    return score

  def r2_fit_and_score(self, model_name: str) -> None:
    model = self.models[model_name]

    model.fit(self.x_train, self.y_train)
    y_pred = model.predict(self.x_test)
    self.r2_scores[model_name] = self.compare_y_pred_y_test(y_pred, model_name)


if __name__ == '__main__':
  pass
