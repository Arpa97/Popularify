import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import random

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats  
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import itertools


global object_cols
object_cols = ['artist_name', 'track_id', 'track_name', 'key_notes','pop_cat']

global numeric_cols
numeric_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness',
        'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence',
       'popularity','pop_frac','pop_bin']

global categorical_cols
categorical_cols = ['key', 'mode', 'time_signature']

global numeric_non_cat
numeric_non_cat = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
       'loudness', 'speechiness', 'tempo','valence',
       'popularity','pop_frac','pop_bin']

global cols_to_standardize
cols_to_standardize = ['duration_ms', 'loudness', 'tempo']

### FUNCTIONS ####

def load_data():
    df = pd.read_csv('SpotifyAudioFeaturesApril2019.csv')
    return df

def set_view_options(max_cols=50, max_rows=50, max_colwidth=9, dis_width=250):
    pd.options.display.max_columns = max_cols
    pd.options.display.max_rows = max_rows
    pd.set_option('max_colwidth', max_colwidth)
    pd.options.display.width = dis_width

def rename_columns(df):
    df.columns = ['artist', 'trk_id', 'trk_name', 'acous', 'dance', 'ms', 
                  'energy', 'instr', 'key', 'live', 'loud', 'mode', 'speech', 
                  'tempo', 't_sig', 'val', 'popularity']
    return df

def get_df_info(df):
    print(df.head())
    print("The columns are:")
    print(df.columns)
    print(df.info())
    print("Do we have any nulls?")
    print(f"Looks like we have {df.isnull().sum().sum()} nulls")
    pop_mean = df['popularity'].mean()
    print(pop_mean)
    print(df[df['popularity'] >= 50 ]['popularity'].count() / df.shape[0])
    print(df['artist_name'].unique().shape)
    print(df['artist_name'].value_counts())
def describe_cols(df, L=10):
    O = pd.get_option("display.max_colwidth")
    pd.set_option("display.max_colwidth", L)
    print(df.rename(columns=lambda x: x[:L - 2] + '...' if len(x) > L else x).describe())
    pd.set_option("display.max_colwidth", O)

def most_popular_songs(df):
    most_popular = df[df['popularity'] > 90]['popularity'].count()
    print(df[df['popularity'] > 90][['artist_name', 'popularity']])

def scatter_plot(df, col_x, col_y):
    plt.scatter(df[col_x], df[col_y], alpha=0.2)
    plt.show()

def plot_scatter_matrix(df, num_rows):
    scatter_matrix(df[:num_rows], alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()

def calc_correlations(df, cutoff):
    corr = df.corr()
    print(corr[corr > cutoff])

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            if df[cols[i]].dtype != 'object' and df[cols[j]].dtype != 'object':
                pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    df = df.copy()
    df.drop(labels=['artist_name','track_id','track_name',], axis=1, inplace=True)
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    
    print("The top absolute correlations are:")
    print(au_corr[0:n])
    return au_corr[0:n]

def linear_regression_initial(df):
    df = df.copy()

    X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
          'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
          'speechiness', 'tempo', 'time_signature', 'valence']
    y_col = ['popularity']
    X = df[X_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    fitted_vals = results.predict(X_train)
    stu_resid = results.resid_pearson
    residuals = results.resid
    y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, \
                           'stu_resid': stu_resid})
    print(results.summary())
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title("QQ Plot - Initial Linear Regression")
    fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)
    plt.show()
    y_vals.plot(kind='scatter', x='fitted_vals', y='stu_resid')
    plt.show()

def get_zeros(df):
    print(df[df['popularity'] == 0 ]['popularity'].count())

def plot_pop_dist(df):
    # set palette
    sns.set_palette('muted')

    # create initial figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    sns.distplot(df['popularity']/100, color='g', label="Popularity").set_title("Distribution of Popularity Scores - Entire Data Set")

    # create x and y axis labels
    plt.xlabel("Popularity")
    plt.ylabel("Density")

    plt.show()

def undersample_plot(df):
    sns.set_palette('muted')
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    sns.distplot(df['popularity']/100, color='g', label="Popularity").set_title("Illustration of Undersampling from Data Set")
    line = ax.get_lines()[-1]
    x_line, y_line = line.get_data()
    mask = x_line > 0.55
    x_line, y_line = x_line[mask], y_line[mask]
    ax.fill_between(x_line, y1=y_line, alpha=0.5, facecolor='red')
    label_x = 0.5
    label_y = 4
    arrow_x = 0.6
    arrow_y = 0.2
    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "Sample all songs in this range.\n Cutoff = 0.5.", xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
        arrowprops=arrow_properties)
    label_x = 0.1
    label_y = 3
    arrow_x = 0.2
    arrow_y = 0.2

    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "Randomly sample n songs", xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='g', alpha=0.5),
        arrowprops=arrow_properties)
    plt.annotate(
        "Resulting dataset with 50/50 \n Popular/Not Popular songs", xy=(0.6, 2),
        xytext=(0.62, 2),
        bbox=dict(boxstyle='round,pad=0.5', fc='b', alpha=0.5))
    plt.xlabel("Popularity")
    plt.ylabel("Density")

    plt.show()

def get_stats(df):
    print(f"There are {df.shape[0]} rows")
    print(f"There are {df['track_id'].unique().shape} unique songs")
    print(f"There are {df['artist_name'].unique().shape} unique artists")
    print(f"There are {df['popularity'].unique().shape} popularity scores")
    print(f"The mean popularity score is {df['popularity'].mean()}")
    print(f"There are {df[df['popularity'] > 55]['popularity'].count()} songs with a popularity score > 55")
    print(f"There are {df[df['popularity'] > 75]['popularity'].count()} songs with a popularity score > 75")
    print(f"Only {(df[df['popularity'] > 80]['popularity'].count() / df.shape[0])*100:.2f} % of songs have a popularity score > 80")

def plot_univ_dists(df, cutoff):
    popularity_cutoff = cutoff
    print('Mean value for Danceability feature for Popular songs: {}'.format(df[df['popularity'] > popularity_cutoff]['danceability'].mean()))
    print('Mean value for Danceability feature for Unpopular songs: {}'.format(df[df['popularity'] < popularity_cutoff]['danceability'].mean()))
    
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    fig.suptitle('Histograms and Univariate Distributions of Important Features')
    sns.distplot(df[df['popularity'] < popularity_cutoff]['danceability'])
    sns.distplot(df[df['popularity'] > popularity_cutoff]['danceability'])
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.distplot(df[df['popularity'] < popularity_cutoff]['valence'])
    sns.distplot(df[df['popularity'] > popularity_cutoff]['valence'])
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.distplot(df[df['popularity'] < popularity_cutoff]['acousticness'])
    sns.distplot(df[df['popularity'] > popularity_cutoff]['acousticness'])
    plt.show()

def plot_keys(df, cutoff):
    df_popular = df[df['popularity'] > cutoff].copy()
    
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8,5))
    key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 4.0: 'E', 5.0: 
                  'F', 6.0: 'F♯,G♭', 7.0: 'G', 8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 
                  11.0: 'B'}
    
    df_popular['key_val'] = df_popular['key'].map(key_mapping)
    sns.countplot(x='key_val', data=df_popular, order=df_popular['key_val'].value_counts().index, palette='muted')
    plt.title("Key Totals for Popular Songs")
    plt.show()

    df_unpopular = df[df['popularity'] < 55].copy()
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8,5))
    df_unpopular['key_val'] = df_unpopular['key'].map(key_mapping)
    sns.countplot(x='key_val', data=df_unpopular, order=df_unpopular['key_val'].value_counts().index, palette='muted')
    plt.title("Key Totals for Unpopular Songs")
    plt.show()

def plot_heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def calc_ANOVA(df, cutoff):
    df_popular = df[df['popularity'] > cutoff].copy()
    df_unpopular = df[df['popularity'] < cutoff].copy()

    print("Popular and Unpopular Danceability Means:")  
    print(df_popular['danceability'].mean())
    print(df_unpopular['danceability'].mean())
    f_val, p_val = stats.f_oneway(df_popular['danceability'], df_unpopular['danceability'])  
    
    print("Danceability One-way ANOVA P ={}".format(p_val)) 

    print("Popular and Unpopular Loudness Means:")  
    print(df_popular['loudness'].mean())
    print(df_unpopular['loudness'].mean())
    f_val, p_val = stats.f_oneway(df_popular['loudness'], df_unpopular['loudness'])  
    
    print("Loudness One-way ANOVA P ={}".format(p_val)) 

    print(df_popular['valence'].mean())
    print(df_unpopular['valence'].mean())
    f_val, p_val = stats.f_oneway(df_popular['valence'], df_unpopular['valence'])  
    
    print("Valence One-way ANOVA P ={}".format(p_val))

    print(df_popular['instrumentalness'].mean())
    print(df_unpopular['instrumentalness'].mean())
    f_val, p_val = stats.f_oneway(df_popular['instrumentalness'], df_unpopular['instrumentalness'])  
    
    print("Instrumentalness One-way ANOVA P ={}".format(p_val))

def random_under_sampler(df, cutoff):
    df_original = df.copy()
    df_original['pop_bin'] = np.where(df_original['popularity'] > cutoff, "Popular", "Not_Popular")

    df_small = df_original[df_original['popularity'] > cutoff].copy()
    df_samples_added = df_small.copy()
    
    total = df_small.shape[0] + 1

    # loop through and add random unpopular rows to sampled df
    while total <= df_small.shape[0]*2:

        # pick a random from from the original dataframe
        rand_row = random.randint(0,df_original.shape[0])
        
        if df_original.loc[rand_row, 'pop_bin'] == "Not_Popular":
            df_samples_added.loc[total] = df_original.loc[rand_row, :]
            total +=1

    # print some stats on the undersampled df
    print("Size checks for new df:")
    print("Shape of new undersampled df: {}".format(df_samples_added.shape))
    print(df_samples_added['pop_bin'].value_counts())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Popular']['danceability'].mean())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Not_Popular']['danceability'].mean())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Popular']['danceability'].count())
    print(df_samples_added[df_samples_added['pop_bin'] == 'Not_Popular']['danceability'].count())
    f_val, p_val = stats.f_oneway(df_samples_added[df_samples_added['pop_bin'] == 'Popular']['danceability'], df_samples_added[df_samples_added['pop_bin'] == 'Not_Popular']['danceability'])  
  
    print("One-way ANOVA P ={}".format(p_val))

    # return the df
    return df_samples_added

def plot_hist(sampled_df):
    sampled_df[sampled_df['pop_bin'] == "Popular"].hist(figsize=(8, 8))  
    plt.show()

    sampled_df[sampled_df['pop_bin'] != "Popular"].hist(figsize=(8, 8))
    plt.show()

def search_artist_track_name(df, artist, track):
    # this displays much better in jupyter
    print(df[(df['artist_name'].str.contains(artist)) & (df['track_name'].str.contains(track))])

    # use this if searching for A$AP rocky (or other artist with $ in the name)
    # df[(df['artist_name'].str.contains("A\$AP Rocky"))]

def add_cols(df, cutoff=55):
    df = df.copy()
    
    # add key_notes mapping key num vals to notes
    key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 
                   4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 
                   8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}
    df['key_notes'] = df['key'].map(key_mapping)
    
    # add columns relating to popularity
    df['pop_frac'] = df['popularity'] / 100
    df['pop_cat'] = np.where(df['popularity'] > cutoff, "Popular", "Not_Popular")
    df['pop_bin'] = np.where(df['popularity'] > cutoff, 1, 0)
    
    return df

def split_sample_combine(df, cutoff=55, col='popularity', rand=None):
    # split out popular rows above the popularity cutoff
    split_pop_df = df[df[col] > cutoff].copy()
    
    # get the leftover rows, the 'unpopular' songs
    df_leftover = df[df[col] < cutoff].copy()
    
    # what % of the original data do we now have?
    ratio = split_pop_df.shape[0] / df.shape[0]
    
    # what % of leftover rows do we need?
    ratio_leftover = split_pop_df.shape[0] / df_leftover.shape[0]
    
    # get the exact # of unpopular rows needed, using a random sampler
    unpop_df_leftover, unpop_df_to_add = train_test_split(df_leftover, \
                                                          test_size=ratio_leftover, \
                                                          random_state = rand)
    
    # combine the dataframes to get total rows = split_pop_df * 2
    # ssc stands for "split_sample_combine"
    ssc_df = split_pop_df._append(unpop_df_to_add).reset_index(drop=True)

    # shuffle the df
    ssc_df = ssc_df.sample(frac=1, random_state=rand).reset_index(drop=True)
    
    # add key_notes mapping key num vals to notes
    key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 
                   4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 
                   8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}
    ssc_df['key_notes'] = ssc_df['key'].map(key_mapping)
    
    # add columns relating to popularity
    ssc_df['pop_frac'] = ssc_df['popularity'] / 100
    ssc_df['pop_cat'] = np.where(ssc_df['popularity'] > cutoff, "Popular", "Not_Popular")
    ssc_df['pop_bin'] = np.where(ssc_df['popularity'] > cutoff, 1, 0)
    
    return ssc_df

def standardize_return_X_y(df, std=True, log=False):
    df = df.copy()
    
    # standardize some columns if std = True
    if std == True:
        for col in cols_to_standardize:
            new_col_name = col + "_std"
            df[new_col_name] = (df[col] - df[col].mean()) / df[col].std()

        X_cols = ['acousticness', 'danceability', 'duration_ms_std', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness_std', 'mode', 
                  'speechiness', 'tempo_std', 'time_signature', 'valence']
    else:
        X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
                  'speechiness', 'tempo', 'time_signature', 'valence']
        
    # if log = True, let's transform y to LOG
    if log == True:
        df['pop_log'] = df['popularity'] / 100
        df['pop_log'] = [0.00000001 if x == 0 else x for x in df['pop_log']]
        df['pop_log'] = [0.99999999 if x == 1 else x for x in df['pop_log']]
        df['pop_log'] = np.log(df['pop_log'] / (1 - df['pop_log']))
        y_col = ['pop_log']
            
    else:
        y_col = ['popularity']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]
    
    return X, y

def linear_regression_final(df, show_plots=True):
    X, y = standardize_return_X_y(df, std=True, log=False)
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    fitted_vals = results.predict(X)
    stu_resid = results.resid_pearson
    residuals = results.resid
    y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, \
                           'stu_resid': stu_resid})
    print(results.summary())    
    if show_plots == True:
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.title("True vs. Predicted Popularity Values - Linear Regression")
        plt.plot(y,alpha=0.2, label="True")
        plt.plot(fitted_vals,alpha=0.5, c='r', label="Predicted")
        plt.ylabel("Popularity")
        plt.legend()
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)
        plt.show()
        y_vals.plot(kind='scatter', y='fitted_vals', x='stu_resid')
        plt.show()

    return results

def my_rmse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return np.sqrt(mse)

#FOR RMSE CALCULATION
def linear_regression_sklearn(df, show_plots=True):
    X, y = standardize_return_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    train_predicted = linear.predict(X_train)
    test_predicted = linear.predict(X_test)
    print('RMSE for training set {}'.format(my_rmse(y_train.values, train_predicted)))
    print('RMSE for test set {}'.format(my_rmse(y_test.values, test_predicted)))
    print('The Coefficients are:')
    print(linear.coef_)
    print('The R^2 values is: {}'.format(linear.score(X_train, y_train)))
    if show_plots == True:
        plt.plot(y_train.reset_index(drop=True), alpha=0.2)
        plt.plot(train_predicted, alpha=0.5, c='r')
        plt.show()

        plt.plot(y_test.reset_index(drop=True), alpha=0.2)
        plt.plot(test_predicted, alpha=0.5, c='r')
        plt.show()


def return_X_y_logistic(df):
    df = df.copy()

    X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
              'speechiness', 'tempo', 'time_signature', 'valence']
    y_col = ['pop_bin']

    X = df[X_cols]
    y = df[y_col]

    return X, y

def return_X_y_logistic_more_cols(df):
    df = df.copy()
    X_cols = ['artist_name','track_id','track_name','acousticness', 'danceability', 'duration_ms', 'energy', 
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
              'speechiness', 'tempo', 'time_signature', 'valence']

    # use 1's and 0's for logistic
    y_col = ['pop_bin']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]

    return X, y

def return_X_y_logistic_sig_only(df):
    df = df.copy()
    X_cols = ['danceability','energy', 
              'instrumentalness', 'loudness']

    # use 1's and 0's for logistic
    y_col = ['pop_bin']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]

    return X, y

def standardize_X_sig_only(X):  
    X = X.copy()
    
    cols = ['loudness']
    for col in cols:
        new_col_name = col + "_std"
        X[new_col_name] = (X[col] - X[col].mean()) / X[col].std()
        
    X_cols = ['danceability','energy', 
              'instrumentalness', 'loudness_std']

    # return the std columns in a dataframe
    X = X[X_cols]
    
    return X

def standardize_X(X):  
    X = X.copy()
    
    for col in cols_to_standardize:
        new_col_name = col + "_std"
        X[new_col_name] = (X[col] - X[col].mean()) / X[col].std()
        
    X_cols = ['acousticness', 'danceability', 'duration_ms_std', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness_std', 'mode', 
                  'speechiness', 'tempo_std', 'time_signature', 'valence']

    # return the std columns in a dataframe
    X = X[X_cols]
    
    return X

def standardize_X_train_test(X_train, X_test):  
    X_train = X_train.copy()
    X_test = X_test.copy()     
    for col in cols_to_standardize:
        new_col_name = col + "_std"
        X_train[new_col_name] = (X_train[col] - X_train[col].mean()) / X_train[col].std()
        X_test[new_col_name] = (X_test[col] - X_test[col].mean()) / X_test[col].std()
    
    X_cols = ['acousticness', 'danceability', 'duration_ms_std', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness_std', 'mode', 
                  'speechiness', 'tempo_std', 'time_signature', 'valence']

    # return the std columns in a dataframe
    X_train_std = X_train[X_cols]
    X_test_std = X_test[X_cols]
    
    return X_train_std, X_test_std

def basic_logistic_regression(df, cutoff=85, rand=0, sig_only=False):
    df = df.copy()

    if sig_only == True:
        X, y = return_X_y_logistic_sig_only(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X_sig_only(X)

    else:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=80, rand=rand))
        X = standardize_X(X)

    X_const = add_constant(X, prepend=True)

    logit_model = Logit(y, X_const).fit()
    
    print(logit_model.summary())

    return logit_model

def logistic_regression_final(df, plot_the_roc=True):
    df = df.copy()
    cutoff = 85
    
    X, y = return_X_y_logistic_more_cols(split_sample_combine(df, cutoff=cutoff, rand=2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

    global df_train_results_log80 
    global df_test_results_log80
    df_train_results_log80 = X_train.join(y_train)
    df_test_results_log80 = X_test.join(y_test)

    # standardize X_train and X_test
    X_train = standardize_X(X_train)
    X_test = standardize_X(X_test)

    X_train = X_train.values
    y_train = y_train.values.ravel()

    X_test = X_test.values
    y_test = y_test.values.ravel()

    global sanity_check
    sanity_check = X_test

    ## Run logistic regression on all the data
    classifier = LogisticRegression()
    # note using .predict_proba() below, which is the probability of each class
    
    #predict values for X_train
    y_predict_train = classifier.fit(X_train,y_train).predict(X_train)
    probs_0and1_train = classifier.fit(X_train,y_train).predict_proba(X_train)
    y_prob_P_train = probs_0and1_train[:,1]

    # predict values for X_test
    y_predict_test = classifier.fit(X_train,y_train).predict(X_test)
    probs_0and1_test = classifier.fit(X_train,y_train).predict_proba(X_test) # yes!
    y_prob_P_test = probs_0and1_test[:,1]

    # calculate metrics needed to use for ROC curve below
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_prob_P_train, pos_label=1)
    auc_train = metrics.roc_auc_score(y_train, y_prob_P_train) # note we are scoring on our training data!

    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_prob_P_test, pos_label=1)
    auc_test = metrics.roc_auc_score(y_test, y_prob_P_test) # note we are scoring on our training data!

    # print some metrics
    print("Train accuracy: {:.2f}".format(accuracy_score(y_train, y_predict_train)))
    print("Test accuracy: {:.2f}".format(accuracy_score(y_test, y_predict_test)))

    print("Train recall: {:.2f}".format(recall_score(y_train, y_predict_train)))
    print("Test recall: {:.2f}".format(recall_score(y_test, y_predict_test)))

    print("Train precision: {:.2f}".format(precision_score(y_train, y_predict_train)))
    print("Test precision: {:.2f}".format(precision_score(y_test, y_predict_test)))

    print("Train auc: {:.2f}".format(auc_train))
    print("Test auc: {:.2f}".format(auc_test))

    global conf_matrix_log80_train
    global conf_matrix_log80_test
    conf_matrix_log80_train = confusion_matrix(y_train, y_predict_train)
    conf_matrix_log80_test = confusion_matrix(y_test, y_predict_test)

    global final_coefs
    global final_intercept
    final_coefs = classifier.fit(X_train,y_train).coef_
    final_intercept = classifier.fit(X_train,y_train).intercept_

    # Back of the envelope calcs to make sure metrics above are correct
    df_train_results_log80 = df_train_results_log80.reset_index(drop=True)
    df_train_results_log80['pop_predict'] = y_prob_P_train

    df_test_results_log80 = df_test_results_log80.reset_index(drop=True)
    df_test_results_log80['pop_predict'] = y_prob_P_test

    df_train_results_log80['pop_predict_bin'] = np.where(df_train_results_log80['pop_predict'] >= 0.5, 1, 0)
    df_test_results_log80['pop_predict_bin'] = np.where(df_test_results_log80['pop_predict'] >= 0.5, 1, 0)
    
    print("Back of the envelope calc for Train Recall")
    print(sum((df_train_results_log80['pop_predict_bin'].values * df_train_results_log80['pop_bin'].values))/ df_train_results_log80['pop_bin'].sum())

    if plot_the_roc == True:
        # Plot the ROC
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
                label='Luck')
        ax.plot(fpr_train, tpr_train, color='b', lw=2, label='Model_Train')
        ax.plot(fpr_test, tpr_test, color='r', lw=2, label='Model_Test')
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.set_title("ROC curve - Cutoff: " + str(cutoff), fontsize=24)
        ax.text(0.05, 0.95, " ".join(["AUC_train:",str(auc_train.round(3))]), fontsize=20)
        ax.text(0.32, 0.7, " ".join(["AUC_test:",str(auc_test.round(3))]), fontsize=20)
        ax.legend(fontsize=24)
        plt.show()

def print_confusion_matrix(df, cutoff=55, rand=0):
    df = df.copy()

    X, y = return_X_y_logistic(split_sample_combine(df, cutoff=80, rand=rand))
    X = standardize_X(X)

    X = X.values
    y = y.values.ravel()

    ## Run logistic regression on all the data
    classifier = LogisticRegression()
    # note using .predict() below, which uses default 0.5 for a binary classifier
    y_pred = classifier.fit(X,y).predict(X) # agh! this uses 0.5 threshold for binary classifier
    y_true = y

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    print("| TN | FP |\n| FN | TP |\n")
    print(cnf_matrix)
    print(f"The accurracy is {accuracy_score(y_true, y_pred)}")
    print(f"The accurracy (check) is {(cnf_matrix[1][1]+ cnf_matrix[0][0])/np.sum(cnf_matrix)}")

def plot_cutoffs_vs_metrics(df):
    df = df.copy()

    df_cols = ['auc', 'accuracy', 'precision', 'recall', 'cutoff', 'type']
    df_metrics = pd.DataFrame(columns = df_cols)
    cutoff_range = [45, 55, 60, 65, 70, 75, 80, 85, 90]
    
    for cutoff in cutoff_range:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=cutoff, rand=0))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        X_train = standardize_X(X_train)
        X_test = standardize_X(X_test)

        X_train = X_train.values
        y_train = y_train.values.ravel()

        X_test = X_test.values
        y_test = y_test.values.ravel()
        
        classifier = LogisticRegression()
        y_predict_train = classifier.fit(X_train, y_train).predict(X_train)
        probs_0and1_train = classifier.fit(X_train,y_train).predict_proba(X_train)
        y_prob_P_train = probs_0and1_train[:,1]
        
        test_metrics = []
        # calculate metrics for JUST train
        test_metrics.append(metrics.roc_auc_score(y_train, y_prob_P_train))
        test_metrics.append(accuracy_score(y_train, y_predict_train))
        test_metrics.append(precision_score(y_train, y_predict_train))
        test_metrics.append(recall_score(y_train, y_predict_train))
        test_metrics.append(int(cutoff))
        test_metrics.append("Test")
        
        df_metrics.loc[cutoff] = test_metrics
        df_metrics = df_metrics.reset_index(drop=True)
        df_metrics["cutoff"] = pd.to_numeric(df_metrics["cutoff"])
        
    # plot metrics vs. popularity score cutoff
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    ax.plot(df_metrics['cutoff'], df_metrics['auc'], color='b', lw=2, label='auc')
    ax.plot(df_metrics['cutoff'], df_metrics['accuracy'], color='r', lw=2, label='accuracy')
    ax.plot(df_metrics['cutoff'], df_metrics['precision'], color='g', lw=2, label='precision')
    ax.plot(df_metrics['cutoff'], df_metrics['recall'], color='y', lw=2, label='recall')

    ax.set_xlabel("Popularity Score Cutoff", fontsize=20)
    ax.set_ylabel("Area (auc) / Rate (others)", fontsize=20)
    ax.set_title("Metrics vs Popularity Score Cutoff Values - Training Dataset:", fontsize=24)
    ax.legend(fontsize=24)
    plt.show()

def plot_confusion_matrix(cm, ax, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    """
    font_size = 24
    p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title,fontsize=font_size)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=16)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=16)
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == 1 and j == 1:
            lbl = "(True Positive)"
        elif i == 0 and j == 0:
            lbl = "(True Negative)"
        elif i == 1 and j == 0:
            lbl = "(False Negative)"
        elif i == 0 and j == 1:
            lbl = "(False Positive)"
        ax.text(j, i, "{:0.2f} \n{}".format(cm[i, j], lbl),
                 horizontalalignment="center", size = font_size,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    ax.set_ylabel('True',fontsize=font_size)
    ax.set_xlabel('Predicted',fontsize=font_size)

def plot_conf_matrix_Train():
    fig = plt.figure(figsize=(12,11))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ["Not Popular","Popular"]
    plot_confusion_matrix(conf_matrix_log80_train, ax, classes=class_names,normalize=True,
                      title='Normalized Confusion Matrix, Train Dataset, threshold = 0.5')
    plt.show()

def plot_conf_matrix_Test():
    fig = plt.figure(figsize=(12,11))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ["Not Popular","Popular"]
    plot_confusion_matrix(conf_matrix_log80_test, ax, classes=class_names,normalize=True,
                      title='Normalized Confusion Matrix, Test Dataset, threshold = 0.5')
    plt.show()

def plot_final_coeffs():
    columns_bar = ['acousticness', 'danceability','duration_ms', 'energy', 'instrumentalness', 
                   'key', 'liveness','loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 
                   'valence']
    df_final_coefs = pd.DataFrame(data = final_coefs, columns = columns_bar)
    df_final_coefs.plot(kind = 'bar', figsize=(10, 5), align='edge')
    plt.show()

def get_true_positives():
    # Songs my test model predicted were popular that are actually popular (true positives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 1) & (df_test_results_log80['pop_bin'] == 1)])

def get_true_negatives():
    # Songs my test model predicted were not popular that are not actually popular (true negatives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 0) & (df_test_results_log80['pop_bin'] == 0)])

def get_false_positives():
    # Songs my testodel predicted were popular that are not actually popular (false positives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 1) & (df_test_results_log80['pop_bin'] == 0)])
    # calculate false positive rate
    df_train_results_log80[(df_train_results_log80['pop_predict_bin'] == 1) & (df_train_results_log80['pop_bin'] == 0)].count() / df_train_results_log80[df_train_results_log80['pop_bin'] == 0].count()

def get_false_negatives():
    # Songs my test model predicted were not popular that are actually popular (false negatives)
    print(df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 0) & (df_test_results_log80['pop_bin'] == 1)])

def sanity_check_test():
    # grab a record from the results dataframe
    sanity_check_loc = df_test_results_log80[(df_test_results_log80['pop_predict_bin'] == 0) & (df_test_results_log80['pop_bin'] == 1)].iloc[0]
    # set the probability that song has a popularity score >=80 = sanity_check_prob
    sanity_check_prob = sanity_check_loc['pop_predict']

    # print these to make sure they make sense
    print(sanity_check_loc)
    print(sanity_check_prob)

    # this record coresponds to the 9th row of X_test within the logistic regression function (I know becuase I looked ;)
    print(sanity_check[9, :])
    
    # grab the standardized variables from X_test
    sanity_check_std_vars = sanity_check[9, :]
    print(sanity_check_std_vars)

    # multiply the standardized variables by the regression coefficients, sum them and add the intercept
    mult_coefs_vars_add_intercept = sum(sanity_check_std_vars*final_coefs.reshape(13)) + final_intercept
    print(mult_coefs_vars_add_intercept)

    # since the log odds = P / 1-P, need to exponentiate this to get to the final predicted probability
    exponentiated = np.exp(mult_coefs_vars_add_intercept)
    print(exponentiated)

    # finally, calculate P, the odds of popular (popularity score >= 80)
    p = exponentiated / (1 + exponentiated)
    print(p)

    # does this equal what we think it should???
    delta_ps = float(p - sanity_check_prob)
    print(f"Delta in p = {delta_ps:.7f}")

if __name__ == "__main__":
    # load data
    df = load_data()
    # set nice view options for terminal viewing
    set_view_options(max_cols=50, max_rows=50, max_colwidth=40, dis_width=250)

    # get basic info from dataset
    print("BASIC INFO")
    get_df_info(df)

    # Take a look at the data with truncated columns
    print("TRUNCATED DATA")
    describe_cols(df, 9)

    # look at top correlations - look into multicollinearity
    print("TOP CORRELATIONS")
    get_top_abs_correlations(df, 10)
    print("CORRELATION PLOTS:")
    scatter_plot(df, 'danceability', 'popularity')
    scatter_plot(df, 'duration_ms', 'popularity')
    scatter_plot(df, 'key', 'popularity')
    scatter_plot(df, 'acousticness', 'popularity')
    plot_univ_dists(df, 85)
    plot_keys(df, 55)
    plot_heatmap(df)
    calc_ANOVA(df, 55)



    




