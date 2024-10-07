import os
import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from src.exception import CustomError
from src.logger import logging
from src.utils import save_pickle_file, load_pickle_file