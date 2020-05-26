from IPython.lib.deepreload import reload as dreload
from auto_ml.settings import EMAIL_HOST_USER
from django.core.mail import EmailMessage
import PIL, os, numpy as np, math, collections, threading, json,random, scipy
import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy
import matplotlib
import IPython, graphviz, sklearn_pandas, sklearn, warnings, pdb
import contextlib
from abc import abstractmethod
from glob import glob, iglob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import chain
from functools import partial
from collections import Iterable, Counter, OrderedDict
from isoweek import Week
from IPython.lib.display import FileLink
from PIL import Image, ImageEnhance, ImageOps
from sklearn import metrics, ensemble, preprocessing
from operator import itemgetter, attrgetter
from pathlib import Path
from distutils.version import LooseVersion
from threading import Thread

from matplotlib import pyplot as plt, rcParams, animation
from ipywidgets import interact, interactive, fixed, widgets
matplotlib.rc('animation', html='html5')
np.set_printoptions(precision=5, linewidth=110, suppress=True)

from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()

def in_ipynb():
    try:
        cls = get_ipython().__class__.__name__
        return cls == 'ZMQInteractiveShell'
    except NameError:
        return False

import tqdm as tq
from tqdm import tqdm_notebook, tnrange

def clear_tqdm():
    inst = getattr(tq.tqdm, '_instances', None)
    if not inst: return
    try:
        for i in range(len(inst)): inst.pop().close()
    except Exception:
        pass

if in_notebook():
    def tqdm(*args, **kwargs):
        clear_tqdm()
        return tq.tqdm(*args, file=sys.stdout, **kwargs)
    def trange(*args, **kwargs):
        clear_tqdm()
        return tq.trange(*args, file=sys.stdout, **kwargs)
else:
    from tqdm import tqdm, trange
    tnrange=trange
    tqdm_notebook=tqdm



from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute._base import SimpleImputer as Imputer
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics



def set_plot_sizes(sml, med, big):
    plt.rc('font', size=sml)          # controls default text sizes
    plt.rc('axes', titlesize=sml)     # fontsize of the axes title
    plt.rc('axes', labelsize=med)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sml)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title

def parallel_trees(m, fn, n_jobs=8):
        return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))

def combine_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)

def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement.
    Parameters:
    -----------
    df: A pandas data frame, that you wish to sample from.
    n: The number of rows you wish to sample.
    Returns:
    --------
    return value: A random sample of n rows of df.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> get_sample(df, 2)
       col1 col2
    1     2    b
    2     3    a
    """
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string or list of strings that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    >>>df2 = pd.DataFrame({'start_date' : pd.to_datetime(['3/11/2000','3/13/2000','3/15/2000']),
                            'end_date':pd.to_datetime(['3/17/2000','3/18/2000','4/1/2000'],infer_datetime_format=True)})
    >>>df2
        start_date  end_date
    0   2000-03-11  2000-03-17
    1   2000-03-13  2000-03-18
    2   2000-03-15  2000-04-01
    >>>add_datepart(df2,['start_date','end_date'])
    >>>df2
        start_Year  start_Month start_Week  start_Day   start_Dayofweek start_Dayofyear start_Is_month_end  start_Is_month_start    start_Is_quarter_end    start_Is_quarter_start  start_Is_year_end   start_Is_year_start start_Elapsed   end_Year    end_Month   end_Week    end_Day end_Dayofweek   end_Dayofyear   end_Is_month_end    end_Is_month_start  end_Is_quarter_end  end_Is_quarter_start    end_Is_year_end end_Is_year_start   end_Elapsed
    0   2000        3           10          11          5               71              False               False                   False                   False                   False               False               952732800       2000        3           11          17      4               77              False               False               False               False                   False           False               953251200
    1   2000        3           11          13          0               73              False               False                   False                   False                   False               False               952905600       2000        3           11          18      5               78              False               False               False               False                   False           False               953337600
    2   2000        3           11          15          2               75              False               False                   False                   False                   False               False               953078400       2000        4           13          1       5               92              False               True                False               True                    False           False               954547200
    """
    if isinstance(fldnames,str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)

def is_date(x): return np.issubdtype(x.dtype, np.datetime64)

def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.
    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category
    """
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def apply_cats(df, trn):
    """Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.
    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values. The category codes are determined by trn.
    trn: A pandas dataframe. When creating a category for df, it looks up the
        what the category's code were in trn and makes those the category codes
        for df.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category {a : 1, b : 2}
    >>> df2 = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['b', 'a', 'a']})
    >>> apply_cats(df2, df)
           col1 col2
        0     1    b
        1     2    a
        2     3    a
    now the type of col is category {a : 1, b : 2}
    """
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(trn[n].cat.categories, ordered=True, inplace=True)

def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    Parameters:
    -----------
    df: The data frame that will be changed.
    col: The column of data to fix by filling in missing data.
    name: The name of the new filled column in df.
    na_dict: A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col2'], 'col2', {})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1   500    2    True
    2     3    2   False
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    """ Changes the column col from a categorical type to it's integer codes.
    Parameters:
    -----------
    df: A pandas dataframe. df[name] will be filled with the integer codes from
        col.
    col: The column you wish to change into the categories.
    name: The column name you wish to insert into df. This column will hold the
        integer codes.
    max_n_cat: If col has more categories than max_n_cat it will not change the
        it to its integer codes. If max_n_cat is None, then col will always be
        converted.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category { a : 1, b : 2}
    >>> numericalize(df, df['col2'], 'col3', None)
       col1 col2 col3
    0     1    a    1
    1     2    b    2
    2     3    a    1
    """
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1

def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe. For each column of df
    which is not in skip_flds nor in ignore_flds, na values are replaced by the
    median value of the column.
    Parameters:
    -----------
    df: The data frame you wish to process.
    y_fld: The name of the response variable
    skip_flds: A list of fields that dropped from df.
    ignore_flds: A list of fields that are ignored during processing.
    do_scale: Standardizes each column in df. Takes Boolean Values(True,False)
    na_dict: a dictionary of na columns to add. Na columns are also added if there
        are any missing values.
    preproc_fn: A function that gets applied to df.
    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.
    subset: Takes a random subset of size subset from df.
    mapper: If do_scale is set as True, the mapper variable
        calculates the values used for scaling of variables during training time (mean and standard deviation).
    Returns:
    --------
    [x, y, nas, mapper(optional)]:
        x: x is the transformed version of df. x will not have the response variable
            and is entirely numeric.
        y: y is the response variable
        nas: returns a dictionary of which nas it created, and the associated median.
        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continuous
        variables which is then used for scaling of during test-time.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> train_cats(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category { a : 1, b : 2}
    >>> x, y, nas = proc_df(df, 'col1')
    >>> x
       col2
    0     1
    1     2
    2     1
    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])
    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())])
    >>>round(fit_transform!(mapper, copy(data)), 2)
    8x4 Array{Float64,2}:
    1.0  0.0  0.0   0.21
    0.0  1.0  0.0   1.88
    0.0  1.0  0.0  -0.63
    0.0  0.0  1.0  -0.63
    1.0  0.0  0.0  -1.46
    0.0  1.0  0.0  -0.63
    1.0  0.0  0.0   1.04
    0.0  0.0  1.0   0.21
    """
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


def start_new_thread(function):
    def decorator(*args, **kwargs):
        t = Thread(target = function, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
    return decorator





def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

def get_nn_mappers(df, cat_vars, contin_vars):
    # Replace nulls with 0 for continuous, "" for categorical.
    for v in contin_vars: df[v] = df[v].fillna(df[v].max()+100,)
    for v in cat_vars: df[v].fillna('#NA#', inplace=True)

    # list of tuples, containing variable and instance of a transformer for that variable
    # for categoricals, use LabelEncoder to map to integers. For continuous, standardize
    cat_maps = [(o, LabelEncoder()) for o in cat_vars]
    contin_maps = [([o], StandardScaler()) for o in contin_vars]




def split_vals(a,n):
    return a[:n].copy(), a[n:].copy()

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m,X_train,y_train,X_valid,y_valid):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
def fx(m,X_valid,y_valid):
    return(m.score(X_valid,y_valid))

def auto_train(a,b,X_train,y_train):
    ''' a and b are the min_ leaf and max_ features respectively'''
    q=RandomForestRegressor(n_jobs=-1, min_samples_leaf=a,max_features=b,oob_score=False)
    q.fit(X_train, y_train)
    return q

def data_trainer(Target_Variable,data_raw,n_valid,date_column=None):
    df_raw=data_raw
    reset_rf_samples()
    ''' This if statement is to reduce the date part'''
    if date_column:
        add_datepart(df_raw,date_column)

    train_cats(df_raw)
    df,y,nas=proc_df(df_raw,Target_Variable)
    n_trn=len(df)-n_valid
    raw_train,raw_valid=split_vals(df_raw,n_trn)
    X_train,X_valid=split_vals(df,n_trn)
    y_train,y_valid=split_vals(y,n_trn)

    ''' The Decider is the rf sampling we will be doing to speed up the process'''
    decider=None
    if len(X_train)>20000:
        decider=20000


    ''' from here we are tuning the parameters '''
    if decider:
        set_rf_samples(decider)
    score=0
    min_leaf_a=0
    max_feature_a=None
    z=None
    list1=[1,3,5,10,25]
    list2=[0.1,0.25,0.5,0.75,0.9,"sqrt","log2",1]
    for leafs in list1:
        for features in list2:
            t=auto_train(a=leafs,b=features,X_train=X_train,y_train=y_train)
            print(fx(t,X_valid=X_valid,y_valid=y_valid),leafs)
            if fx(t,X_valid=X_valid,y_valid=y_valid)>score:
                score=fx(m=t,X_valid=X_valid,y_valid=y_valid)
                min_leaf_a=leafs
                max_feature_a=features
                z=t

    ''' from here we are doing the feature engineering'''
    print(min_leaf_a)
    reset_rf_samples()
    z=RandomForestRegressor(n_jobs=-1,min_samples_leaf= min_leaf_a,max_features= max_feature_a,oob_score=False,n_estimators=40)
    z.fit(X_train,y_train)
    fi=rf_feat_importance(z,df)
    score=0
    final_feature_importance_value=0
    feature_importance_value_list=[0,0.001,0.002,0.0025,0.003,0.0035]
    for feature_importance_value in feature_importance_value_list:
        to_keep = fi[fi.imp>=feature_importance_value].cols
        df_keep = df[to_keep].copy()
        X_train, X_valid = split_vals(df_keep, n_trn)
        feature_tuner=RandomForestRegressor(n_jobs=-1,min_samples_leaf= min_leaf_a,max_features= max_feature_a,oob_score=False,n_estimators=40)
        feature_tuner.fit(X_train,y_train)
        if fx(feature_tuner,X_valid=X_valid,y_valid=y_valid)>=score:
            score=fx(feature_tuner,X_valid=X_valid,y_valid=y_valid)
            final_feature_importance_value=feature_importance_value
    '''Here we will be doing the final model training that is for now, with time i will add more tuning'''
    to_keep = fi[fi.imp>=final_feature_importance_value].cols
    df_keep = df[to_keep].copy()
    X_train, X_valid = split_vals(df_keep, n_trn)
    z=RandomForestRegressor(n_jobs=-1,min_samples_leaf= min_leaf_a,max_features= max_feature_a,oob_score=False,n_estimators=40)
    z.fit(X_train,y_train)


    ''' Here we are trying to remove the variables which are temporally dependant or have too many categories'''
    df_ext = df_keep.copy()
    df_ext['is_valid'] = 1
    df_ext.is_valid[:n_trn] = 0
    x, y, nas = proc_df(df_ext, 'is_valid')

    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=5, max_features=0.75, n_jobs=-1, oob_score=True)
    m.fit(x, y);

    if m.oob_score_>0.75:
        fiq = rf_feat_importance(m, x)
        to_drop = fiq[fiq.imp>0.4].cols
        x_2=x.drop(to_drop,axis=1)
        m = RandomForestClassifier(n_estimators=40, min_samples_leaf=5, max_features=0.75, n_jobs=-1, oob_score=True)
        m.fit(x_2, y);
        if m.oob_score_>0.75:
            fiq = rf_feat_importance(m, x_2)
            to_keeps=fiq[fiq.imp<=0.4].cols
            df_kept=df_keep[to_keeps].copy()
            X_train, X_valid = split_vals(df_kept, n_trn)
            aq=RandomForestRegressor(n_jobs=-1,min_samples_leaf=5,max_features=0.5,n_estimators=40)
            aq.fit(X_train,y_train)

        else:
            to_keep=fiq[fiq.imp<=0.4].cols
            df_kept=df_keep[to_keep].copy()
            X_train, X_valid = split_vals(df_kept, n_trn)
            aq=RandomForestRegressor(n_jobs=-1,min_samples_leaf=5,max_features=0.5,n_estimators=40)
            aq.fit(X_train,y_train)
        print_score(aq,X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid)
        print(min_leaf_a,max_feature_a,final_feature_importance_value)


    else:
        z=RandomForestRegressor(n_jobs=-1,min_samples_leaf= min_leaf_a,max_features= max_feature_a,oob_score=False,n_estimators=40)
        z.fit(X_train,y_train)
        print_score(z,X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid)

    return[min_leaf_a,max_feature_a,to_keep]



def auto_applyer(leaf_value,feature_value,feature_list,df_raw1,df_test,target_column,date_column=None):
    reset_rf_samples()

    if date_column:
        if date_column in df_test:
            add_datepart(df_test,date_column)
        if date_column in df_raw1:
            add_datepart(df_raw1,date_column)
    '''First we will pre process both test and raw data'''
    train_cats(df_raw1)
    apply_cats(df=df_test,trn=df_raw1)
    X,y,nas=proc_df(df_raw1,target_column)
    X_test,_,nas = proc_df(df_test, na_dict=nas)
    X=X[feature_list]
    X_test=X_test[feature_list]
    z=RandomForestRegressor(n_jobs=-1,min_samples_leaf=leaf_value,max_features=feature_value,oob_score=False,n_estimators=75)
    z.fit(X,y)
    fi = rf_feat_importance(z,X)
    graphed=fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
    fig_save = graphed.get_figure()
    fig_save.savefig('Feature Importance.png')
    print(z.predict(X_test))
    return z.predict(X_test)


def auto_predictor(Target_Variable,data_raw,n_valid,data_to_predict,date_column=None):
    if date_column:

        data_raw['{}'.format(date_column)]= pd.to_datetime(data_raw['{}'.format(date_column)])
        data_to_predict['{}'.format(date_column)]= pd.to_datetime(data_to_predict['{}'.format(date_column)])
        intermed=data_trainer(Target_Variable=Target_Variable,data_raw=data_raw,n_valid=n_valid,date_column=date_column)
        return(auto_applyer(leaf_value=intermed[0],feature_value=intermed[1],feature_list=intermed[2],df_raw1=data_raw,df_test=data_to_predict,target_column=Target_Variable,date_column=date_column))
    else:
        intermed=data_trainer(Target_Variable=Target_Variable,data_raw=data_raw,n_valid=n_valid)
        return(auto_applyer(leaf_value=intermed[0],feature_value=intermed[1],feature_list=intermed[2],df_raw1=data_raw,df_test=data_to_predict,target_column=Target_Variable))

@start_new_thread
def send_mail_final(target_variable,df_train,df_test,email_id,message,date_column=None):

    try:
        df_pred=auto_predictor(Target_Variable=target_variable,data_raw=df_train,n_valid=int(0.1*len(df_train)),data_to_predict=df_test,date_column=date_column)
        df_pred=pd.DataFrame(df_pred)
        df_pred.to_csv('predictions.csv')
        recepient = email_id
        email=EmailMessage('Processed Data',message,EMAIL_HOST_USER,[recepient])
        email.attach_file('predictions.csv')
        email.attach_file('Feature Importance.png')
        xyz=1

    except:
        message='Something went wrong ! Please try again after checking all the fields.'
        recepient = email_id
        email=EmailMessage('Processed Data',message,EMAIL_HOST_USER,[recepient])
        xyz=0

    if xyz :
        os.remove('predictions.csv')
        os.remove('Feature Importance.png')




        #email.attach(files_send.name, files_send.read(), files_send.content_type)
    email.send()
    return



