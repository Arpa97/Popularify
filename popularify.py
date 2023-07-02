import os
import spotipy
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot

from scipy import stats
import numpy as np


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
def set_view_options(max_cols=50, max_rows=50, max_colwidth=9, dis_width=250):
    pd.options.display.max_columns = max_cols
    pd.options.display.max_rows = max_rows
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
    show_correlations(df, 'popularity', 10)
    '''
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
    '''
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
    #TODO:
        #basic_logistic_regression(df, cutoff=80, rand=0)
        #logistic_regression_with_kfold(df, cutoff=80, rand=0)
        #logistic_regression_with_kfold(df, cutoff=80, rand=0, sig_only=True)
        #print_confusion_matrix(df, cutoff=80, rand=0)