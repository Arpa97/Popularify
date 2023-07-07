import os
import spotipy
import re
import csv
import math
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import gpboost as gpb

from scipy import stats
import numpy as np


global cols_to_standardize
cols_to_standardize = ['duration_ms', 'loudness', 'tempo']


def my_rmse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return np.sqrt(mse)

def spotyLog():
    load_dotenv()
    CLIENT_ID = os.getenv("CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

    # authenticate
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )

    # create spotify session object
    session = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    return session

def load_dataset():
    df = pd.read_csv('SpotifyAudioFeaturesNov2018.csv')
    return df

def predict_popularity(model, songURI, hasConst=False, castToString=True):
    track = session.track(songURI)
    uriFinal="spotify:track:"+songURI
    track_features = session.audio_features(uriFinal)
    popularity = track["popularity"]
    df_song = pd.DataFrame(track_features)
    df_song_features = df_song[['acousticness', 'danceability', 'duration_ms', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
                  'speechiness', 'tempo', 'time_signature', 'valence']]
    if hasConst == True:
        df_song_features.insert(0, 'const', 1.0)
    
    print(df_song_features)
    #faccio la previsione
    prediction = model.predict(df_song_features)
    #stampo la previsione
    if castToString == True:
        print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
    else:
        print("PREDICTED POPULARITY: "+str(prediction))
    #stampo la popolarità della canzone
    print(popularity)
    return prediction

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

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

def show_correlations(df, target, n):
    df = df.copy()
    #drop not valuable columns
    df.drop(['artist_name', 'track_id', 'track_name', 'key', 'mode', 'time_signature'], axis=1, inplace=True)
    #drop null and '' values
    df.drop(df[df.isnull().any(axis=1)].index, inplace=True)
    df.drop(df[df.eq('None').any(axis=1)].index, inplace=True)
    labels = ['acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','popularity']
    
    # get correlations of all features with target
    corrmat = df.corr()
    # get the most correlated features
    top_corr_features = corrmat.index[abs(corrmat[target]) >= 0.5]
    plt.figure(figsize=(10, 10))
    # plot heat map
    fig, ax = plt.subplots()
    
    im, cbar = heatmap(df.corr(), labels, labels, ax=ax,
                       cmap="YlGn", cbarlabel="Correlation")
    texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    
    fig.tight_layout()

    plt.show()

# set some display options so easier to view all columns at once
def set_view_options(max_cols=50, max_colwidth=9, dis_width=250):
    pd.options.display.max_columns = max_cols
    pd.options.display.max_rows = None
    pd.set_option('max_colwidth', max_colwidth)
    pd.options.display.width = dis_width

def print_plot(df, target, feature, plot_type):
    #plot_type = 'scatter' or 'box'
    if plot_type == 'scatter':
        df.plot.scatter(x=feature, y=target, figsize=(10, 5), alpha=0.5)
    elif plot_type == 'box':
        df.boxplot(column=feature, by=target, figsize=(10, 5))
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()
    elif plot_type == '' and feature == '':
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

def linear_regression_initial(df):
    df = df.copy()
    #drop unnecessary columns
    df.drop(['artist_name', 'track_id', 'track_name'], axis=1, inplace=True)
    #split dataset into train and test
    y = df['popularity']
    X = df[['acousticness', 'danceability', 'duration_ms', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
                  'speechiness', 'tempo', 'time_signature', 'valence']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = sm.add_constant(X_train)

   # Instantiate OLS model, fit, predict, get errors
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    fitted_vals = results.predict(X_train)
    stu_resid = results.resid_pearson
    residuals = results.resid
    y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, \
                           'stu_resid': stu_resid})
    
    print(results.summary())
    # QQ Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title("QQ Plot - Initial Linear Regression")
    fig = sm.qqplot(stu_resid, line='45', fit=True, ax=ax)
    plt.show()

    # Residuals Plot
    y_vals.plot(kind='scatter', x='fitted_vals', y='stu_resid')
    plt.show()
    return results

def show_undersampling(df):
    # set palette
    sns.set_palette('muted')

    # create initial figure
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    sns.distplot(df['popularity']/100, color='g', label="Popularity").set_title("Illustration of Undersampling from Data Set")
    
    # create line to shade to the right of
    line = ax.get_lines()[-1]
    x_line, y_line = line.get_data()
    mask = x_line > 0.55
    x_line, y_line = x_line[mask], y_line[mask]
    ax.fill_between(x_line, y1=y_line, alpha=0.5, facecolor='red')

    # get values for and plot first label
    label_x = 0.5
    label_y = 4
    arrow_x = 0.6
    arrow_y = 0.2

    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "Per cominciare, creiamo un campione di canzoni \n in questo intervallo ottenuto utilizzando \n un valore di cutoff di 0.5",
        xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
        arrowprops=arrow_properties)

    # Get values for and plot second label
    label_x = 0.1
    label_y = 3
    arrow_x = 0.2
    arrow_y = 0.2

    arrow_properties = dict(
        facecolor="black", width=2,
        headwidth=4,connectionstyle='arc3,rad=0')

    plt.annotate(
        "Successivamente, campioniamo casualmente \n n canzoni in questo intervallo", 
        xy=(arrow_x, arrow_y),
        xytext=(label_x, label_y),
        bbox=dict(boxstyle='round,pad=0.5', fc='g', alpha=0.5),
        arrowprops=arrow_properties)

    # plot final word box
    plt.annotate(
        "Così facendo, otteniamo un \n campione con 50/50 possibilità \n che un brano sia popolare", xy=(0.6, 2),
        xytext=(0.62, 2),
        bbox=dict(boxstyle='round,pad=0.5', fc='b', alpha=0.5))

    # create x and y axis labels
    plt.xlabel("Popularity")
    plt.ylabel("Density")

    plt.show()

def get_stats(df):
    # get stats for each feature
    print(f"There are {df.shape[0]} rows")
    print(f"There are {df['track_id'].unique().shape} unique songs")
    print(f"There are {df['artist_name'].unique().shape} unique artists")
    print(f"There are {df['popularity'].unique().shape} popularity scores")
    print(f"The mean popularity score is {df['popularity'].mean()}")
    print(f"There are {df[df['popularity'] > 55]['popularity'].count()} songs with a popularity score > 55")
    print(f"There are {df[df['popularity'] > 75]['popularity'].count()} songs with a popularity score > 75")
    print(f"Only {(df[df['popularity'] > 80]['popularity'].count() / df.shape[0])*100:.2f} % of songs have a popularity score > 80")
# check that deltas in means are significant for selected dependent variables
def calculate_ANOVA(df, cutoff):
    df_popular = df[df['popularity'] > cutoff].copy()
    df_unpopular = df[df['popularity'] <= cutoff].copy()
    print("Medie di danceability per canzoni popolari e non popolari:")  
    print(df_popular['danceability'].mean())
    print(df_unpopular['danceability'].mean())
    f_val, p_val = stats.f_oneway(df_popular['danceability'], df_unpopular['danceability'])
    print("Danceability One-way ANOVA P =", p_val)
    print("Medie di energy per canzoni popolari e non popolari:")
    print(df_popular['energy'].mean())
    print(df_unpopular['energy'].mean())
    f_val, p_val = stats.f_oneway(df_popular['energy'], df_unpopular['energy'])
    print("Energy One-way ANOVA P =", p_val)
    print("Medie di loudness per canzoni popolari e non popolari:")
    print(df_popular['loudness'].mean())
    print(df_unpopular['loudness'].mean())
    f_val, p_val = stats.f_oneway(df_popular['loudness'], df_unpopular['loudness'])
    print("Loudness One-way ANOVA P =", p_val) 
    print("Medie di valence per canzoni popolari e non popolari:")
    print(df_popular['valence'].mean())
    print(df_unpopular['valence'].mean())
    f_val, p_val = stats.f_oneway(df_popular['valence'], df_unpopular['valence'])
    print("Valence One-way ANOVA P =", p_val)
    print("Medie di strumentalità per canzoni popolari e non popolari:")
    print(df_popular['instrumentalness'].mean())
    print(df_unpopular['instrumentalness'].mean())
    f_val, p_val = stats.f_oneway(df_popular['instrumentalness'], df_unpopular['instrumentalness'])
    print("Instrumentalness One-way ANOVA P =", p_val)
    print("Medie di BPM per canzoni popolari e non popolari:")
    print(df_popular['tempo'].mean())
    print(df_unpopular['tempo'].mean())
    f_val, p_val = stats.f_oneway(df_popular['tempo'], df_unpopular['tempo'])
    print("Tempo One-way ANOVA P =", p_val)

def under_sampler(df, cutoff):
    df = df.copy()
    # get number of popular songs
    num_popular = df[df['popularity'] > cutoff].shape[0]
    # get number of unpopular songs
    num_unpopular = df[df['popularity'] <= cutoff].shape[0]
    # get difference between popular and unpopular songs
    diff = num_unpopular - num_popular
    # get random sample of unpopular songs
    df_unpopular = df[df['popularity'] <= cutoff].sample(n=num_popular, random_state=42)
    # concatenate popular and unpopular songs
    df_under = pd.concat([df_unpopular, df[df['popularity'] > cutoff]])
    # shuffle data
    df_under = df_under.sample(frac=1, random_state=42)

     # print some stats on the undersampled df
    print("Size checks for new df:")
    print("Shape of new undersampled df: {}".format(df_under.shape))
    print(df_under[df_under['popularity'] > cutoff]['danceability'].mean())
    print(df_under[df_under['popularity'] < cutoff]['danceability'].mean())
    print(df_under[df_under['popularity'] > cutoff]['danceability'].count())
    print(df_under[df_under['popularity'] < cutoff]['danceability'].count())
    f_val, p_val = stats.f_oneway(df_under[df_under['popularity'] > cutoff]['danceability'], df_under[df_under['popularity'] < cutoff]['danceability'])  
  
    print("One-way ANOVA P ={}".format(p_val))


    return df_under

def add_cols(df, cutoff):
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

def split_sample(df, cutoff = 55, col = 'popularity', rand=None):
   # choose cutoff, sample popular data, randomly sample unpopular data, and combine the dfs
    df = df.copy()
    df_popular = df[df[col] > cutoff]
    df_unpopular = df[df[col] <= cutoff].sample(n=df_popular.shape[0], random_state=rand)
    df_sample = pd.concat([df_popular, df_unpopular])
    df_sample = df_sample.sample(frac=1, random_state=rand)

    return df_sample

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
    #ssc_df = split_pop_df.append(unpop_df_to_add).reset_index(drop=True)
    ssc_df = pd.concat([split_pop_df, unpop_df_to_add]).reset_index(drop=True)
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
        for col in ['duration_ms', 'loudness', 'tempo']:
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

def linear_regression_final(df):
    df = df.copy()

    #X, y = standardize_return_X_y(df, std=False, log=False)
    X = df[['acousticness', 'danceability', 'duration_ms', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
                  'speechiness', 'tempo', 'time_signature', 'valence']]
    y = df['popularity']

    # Add constant
    X = sm.add_constant(X)

    # Instantiate OLS model, fit, predict, and get errors
    model = sm.OLS(y, X)
    results = model.fit()
    fitted_vals = results.predict(X)
    stu_resid = results.resid_pearson
    residuals = results.resid
    y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, \
                           'stu_resid': stu_resid})

    # Maybe do a line graph for this?
    print(results.summary())
    
    ### Plot predicted values vs. actual/true
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title("True vs. Predicted Popularity Values - Initial Linear Regression")
    plt.plot(y,alpha=0.2, label="True")
    plt.plot(fitted_vals,alpha=0.5, c='r', label="Predicted")
    plt.ylabel("Popularity")
    plt.legend()
    plt.show()

    return results

def linear_regression_sklearn(df):

    df = df.copy()

    #X, y = standardize_return_X_y(df, std=False, log=False)
    X = df[['acousticness', 'danceability', 'duration_ms', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
                  'speechiness', 'tempo', 'time_signature', 'valence']]
    y = df['popularity']

    # Add constant
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Fit model using the training set
    linear = LinearRegression()
    linear = linear.fit(X_train, y_train)

    # Call predict to get the predicted values for training and test set
    train_predicted = linear.predict(X_train)
    test_predicted = linear.predict(X_test)

    # Calculate RMSE for training and test set
    print('RMSE for training set {}'.format(my_rmse(y_train.values, train_predicted)))
    print('RMSE for test set {}'.format(my_rmse(y_test.values, test_predicted)))
    print('The Coefficients are:')
    print(linear.coef_)
    print('The R^2 values is: {}'.format(linear.score(X_train, y_train)))
    return linear

def return_X_y_logistic(df):
    df = df.copy()

    # define columns to use for each
    X_cols = ['acousticness', 'danceability', 'duration_ms', 'energy', 
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 
              'speechiness', 'tempo', 'time_signature', 'valence']

    # use 1's and 0's for logistic
    y_col = ['pop_bin']

    # split into X and y
    X = df[X_cols]
    y = df[y_col]

    return X, y

def return_X_y_logistic_more_cols(df):
    df = df.copy()

    # define columns to use for each
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

    # define columns to use for each
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
    # standardize only columns not between 0 and 1
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
    
    # standardize only columns not between 0 and 1
    for col in cols_to_standardize:
        new_col_name = col + "_std"
        if not math.isnan(X[col].std())  and X[col].std() != 0:
            X[new_col_name] = (X[col] - X[col].mean()) / X[col].std()
            #print("DA VALUES:   " , X[col], " ", X[col].mean(), " ", X[col].std())
        else:
            X[new_col_name] = X[col]
            print("skipped: ", col)
            
        
        
    X_cols = ['acousticness', 'danceability', 'duration_ms_std', 'energy', 
                  'instrumentalness', 'key', 'liveness', 'loudness_std', 'mode', 
                  'speechiness', 'tempo_std', 'time_signature', 'valence']

    # return the std columns in a dataframe
    X = X[X_cols]
    
    return X

def standardize_X_train_test(X_train, X_test):  
    X_train = X_train.copy()
    X_test = X_test.copy() 
    
    # standardize only columns not between 0 and 1
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

# Create a basic logistic regression
def basic_logistic_regression(df, cutoff=55, rand=0, sig_only=False):
    df = df.copy()

    if sig_only == True:
        X, y = return_X_y_logistic_sig_only(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X_sig_only(X)

    else:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=80, rand=rand))
        X = standardize_X(X)

    X_const = sm.add_constant(X, prepend=True)

    logit_model = Logit(y, X_const).fit()
    
    print(logit_model.summary())

    return logit_model

def logistic_regression_with_kfold(df, cutoff=55, rand=0, sig_only=False):
    df = df.copy()
    
    if sig_only == True:
        X, y = return_X_y_logistic_sig_only(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X_sig_only(X)

    else:
        X, y = return_X_y_logistic(split_sample_combine(df, cutoff=cutoff, rand=rand))
        X = standardize_X(X)

    X = X.values
    y = y.values.ravel()

    classifier = LogisticRegression()

    # before kFold
    y_predict = classifier.fit(X, y).predict(X)
    y_true = y
    accuracy_score(y_true, y_predict)
    print(f"accuracy: {accuracy_score(y_true, y_predict)}")
    print(f"precision: {precision_score(y_true, y_predict)}")
    print(f"recall: {recall_score(y_true, y_predict)}")
    print(f"The coefs are: {classifier.fit(X,y).coef_}")

    # with kfold
    kfold = KFold(len(y),shuffle=False)

    accuracies = []
    precisions = []
    recalls = []

    

    for train_index, test_index in kfold.split(X):
        model = LogisticRegression()
        model.fit(X[train_index], y[train_index])

        y_predict = model.predict(X[test_index])
        y_true = y[test_index]

        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))

    print(f"accuracy: {np.average(accuracies)}")
    print(f"precision: {np.average(precisions)}")
    print(f"recall: {np.average(recalls)}")

# this is the code for the final logistic regression I chose, after running all the above
# logistic regression models and k-fold cross-val analysis
def logistic_regression_final(df, plot_the_roc=True, cutoff = 80):
    df = df.copy()
    
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
    return X_train, y_train ,X_test, y_test

# print out confusion matrix
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

# plot popularity score cutoffs vs. logistic regression metrics
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

# plot a confusion matrix
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

# plot confusion matrix for final Train dataset
def plot_conf_matrix_Train():
    fig = plt.figure(figsize=(12,11))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ["Not Popular","Popular"]
    plot_confusion_matrix(conf_matrix_log80_train, ax, classes=class_names,normalize=True,
                      title='Normalized Confusion Matrix, Train Dataset, threshold = 0.5')
    plt.show()

# plot confusion matrix for final Test dataset
def plot_conf_matrix_Test():
    fig = plt.figure(figsize=(12,11))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ["Not Popular","Popular"]
    plot_confusion_matrix(conf_matrix_log80_test, ax, classes=class_names,normalize=True,
                      title='Normalized Confusion Matrix, Test Dataset, threshold = 0.5')
    plt.show()

# plot final coefficients of logistic regression
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
    print(f"Dela in p values is {delta_ps:.7f}, woo hoo!!!")

def return_X_y_GLMM(df):
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

def predict_popularity_logistic(X_train, y_train, X_test, y_test,PLAYLIST_LINK = "https://open.spotify.com/playlist/37i9dQZF1DWSxF6XNtQ9Rg?si=efc4b6ea5d79470a",cutoff=70):

    load_dotenv()
    #save your credentials in a .env file using the following format
        #CLIENT_ID=your_client_id
        #CLIENT_SECRET=your_client_secret
    #to replicate the results.
    CLIENT_ID = os.getenv("CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
    OUTPUT_FILE_NAME = "track_info.csv"
    

    # authenticate
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )

    # create spotify session object
    session = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # get uri from https link
    if match := re.match(r"https://open.spotify.com/playlist/(.*)\?", PLAYLIST_LINK):
        playlist_uri = match.groups()[0]
    else:
        raise ValueError("Expected format: https://open.spotify.com/playlist/...")

    # get list of tracks in a given playlist (note: max playlist length 100)
    tracks = session.playlist_tracks(playlist_uri)["items"]

    # create csv file
    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # write header column names
        writer.writerow(["track_name", "artist_name","track_id","popularity"])

        # extract name, artist and track id from each track
        for track in tracks:
            name = track["track"]["name"]
            artists = ", ".join(
                [artist["name"] for artist in track["track"]["artists"]]
            )
            track_id = track["track"]["id"]
            popularity = track["track"]["popularity"]

            # write to csv
            writer.writerow([name, artists, track_id,popularity])

    # read csv file
    df = pd.read_csv(OUTPUT_FILE_NAME)
    #order by popularity
    df = df.sort_values(by=['popularity'], ascending=False)
    ##take top 10
    #df = df.head(10)
    #add audio features
    audio_features = []
    for index, row in df.iterrows():
        audio_features.append(session.audio_features(row['track_id']))
    df["danceability"]=[x[0]['danceability'] for x in audio_features]
    df["energy"]= [x[0]['energy'] for x in audio_features]
    df["loudness"]= [x[0]['loudness'] for x in audio_features]
    df["speechiness"]= [x[0]['speechiness'] for x in audio_features]
    df["acousticness"]= [x[0]['acousticness'] for x in audio_features]
    df["instrumentalness"]= [x[0]['instrumentalness'] for x in audio_features]
    df["liveness"]= [x[0]['liveness'] for x in audio_features]
    df["valence"]= [x[0]['valence'] for x in audio_features]
    df["tempo"]= [x[0]['tempo'] for x in audio_features]
    df["duration_ms"]= [x[0]['duration_ms'] for x in audio_features]
    df["time_signature"]= [x[0]['time_signature'] for x in audio_features]
    df["key"]= [x[0]['key'] for x in audio_features]
    df["mode"]= [x[0]['mode'] for x in audio_features]
        #add audio features to dataframe    
        
    df_input = df[["artist_name", "track_id", "track_name", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "time_signature", "valence", "popularity"]]
    df_input = add_cols(df_input,cutoff)
    #create y to give to the predict function using df_input data
    #create X to give to the predict function using df_input data
    X,y = return_X_y_logistic_more_cols(df_input)
    X=standardize_X(X)
    X=X.values

    ## Run logistic regression on all the data
    classifier = LogisticRegression()
    # note using .predict_proba() below, which is the probability of each class

    # predict values for X
    y_predict = classifier.fit(X_train,y_train).predict(X)
    probs_0and1 = classifier.fit(X_train,y_train).predict_proba(X) # yes!
    y_prob_P = probs_0and1[:,1]

    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y, y_prob_P, pos_label=1)
    #auc_test = metrics.roc_auc_score(y, y_prob_P_test) # note we are scoring on our training data!

    # print some metrics
    print("Test accuracy: {:.2f}".format(accuracy_score(y, y_predict)))

    print("Test recall: {:.2f}".format(recall_score(y, y_predict)))

    print("Test precision: {:.2f}".format(precision_score(y, y_predict)))

    print("Results on test set:", "\n", probs_0and1)
    
    df_input['pop_predict'] = y_prob_P
    df_input['pop_pred_bin']=np.where(df_input['pop_predict'] >= 0.5, 1, 0)
    df_results = df_input[['artist_name','track_name','popularity','pop_predict','pop_bin','pop_pred_bin']]

    
    print(df_results)
    #print("Test auc: {:.2f}".format(auc_test))



if __name__ == '__main__':
    #LOGIN TO SPOTIFY-API and CREATE A NEW SESSION TO ACCESS THE API
    #this will be used later to test with real songs our model
    session = spotyLog() 
    #load the dataset
    df = load_dataset()
    #setting some display options to make it easier to view all columns at once
    set_view_options()#leaving default values.

    #let's start analyzing the dataset
    #top absolute correlations with popularity
    '''
    show_correlations(df, 'popularity', 10)

    #plotting correlation between popularity and other features
    print_plot(df, 'popularity', 'danceability' ,'scatter')
    print_plot(df, 'popularity', 'energy' ,'scatter')
    print_plot(df, 'popularity', 'loudness' ,'scatter')
    print_plot(df, 'popularity', 'speechiness' ,'scatter')
    print_plot(df, 'popularity', 'acousticness' ,'scatter')
    print_plot(df, 'popularity', 'instrumentalness' ,'scatter')
    print_plot(df, 'popularity', 'liveness' ,'scatter')
    print_plot(df, 'popularity', 'valence' ,'scatter')
    print_plot(df, 'popularity', 'tempo' ,'scatter')
    print_plot(df, 'popularity', 'duration_ms' ,'scatter')
    
    #FIRST APPROACH: Linear Regression
    linearWhole = linear_regression_initial(df)

    #applying undersampling to balance the dataset
    #print_plot(df, 'popularity', '' ,'')
    show_undersampling(df)
    #get_stats(df)
    #calculate_ANOVA(df, 55)
    #actually making a random undersampling of the dataset
    df_samples = under_sampler(df, 80)

    #linear regression with undersampled dataset
    linearModelI = linear_regression_initial(df_samples)

    df_cols = add_cols(df, 80)
    df_split = split_sample(df, cutoff=65, rand=0)

    linearModelF=linear_regression_final(df_split)

    linearModelSk = linear_regression_sklearn(df_split)

    #using linearModel to predict popularity of a song
    predict_popularity(linearModelF, '4Li2WHPkuyCdtmokzW2007', hasConst=True)
    predict_popularity(linearModelI, '4Li2WHPkuyCdtmokzW2007', hasConst=True)
    predict_popularity(linearModelSk, '4Li2WHPkuyCdtmokzW2007', hasConst=True, castToString=False)
    predict_popularity(linearWhole, '4Li2WHPkuyCdtmokzW2007', hasConst=True)
    ###LINEAR MODEL RESULTS WERE NOT GOOD ENOUGH, LET'S TRY LOGISTIC REGRESSION### 
'''
    #df= add_cols(df, 80)
    #logFin = basic_logistic_regression(df, cutoff=80, rand=0)
    #logistic_regression_with_kfold(df, cutoff=80, rand=0)
    #logistic_regression_with_kfold(df, cutoff=80, rand=0, sig_only=True)
    #print_confusion_matrix(df, cutoff=80, rand=0)
    #X_train, y_train, X_test, y_test = logistic_regression_final(df, plot_the_roc=False)
    #print(final_coefs)
    #get_true_positives()
    #get_true_negatives()
    #get_false_positives()
    #get_false_negatives()
    
    #return a new df with songs from a playlist. 
    #predict_popularity_logistic(X_train, y_train, X_test, y_test, cutoff=50)
    #predict_popularity_logistic(X_train, y_train, X_test, y_test, cutoff=80, PLAYLIST_LINK='https://open.spotify.com/playlist/37i9dQZF1DWXRPjCBAuFj3?si=b7eea88649c045b1')
    
    #Not accurate enaugh, let's try with our last model: Mixed Model

    starting_GLMM(df, cutoff=80)
        