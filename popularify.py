import os
import numpy as np 
import pandas as pd # for working with dataframes
import seaborn as sns # for data visualization 
import joblib # for saving models

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials


from matplotlib import pyplot as plt # for plotting
from pandas.plotting import table # for plotting tables
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")



dataframe = pd.read_csv('SpotifyAudioFeaturesApril2019.csv')
print(dataframe.head())

print("Dataframe description: \n")
des = dataframe.describe()
print (des.keys())
with pd.option_context('display.max_seq_items', None):
    print(des)
#saving des as .csv
np.savetxt("dataframe_description.csv", des, delimiter=",", fmt='%s')

plt.show()
print("Dataframe keys: \n")
print(dataframe.keys())

print(pd.isnull(dataframe).sum())

#drop null rows
dataframe = dataframe.dropna()
#check if null rows are dropped
print(pd.isnull(dataframe).sum())

#standardize converting key value to corrispondent key signature
key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 
                4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 
                8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}
dataframe['key_string'] = dataframe['key'].map(key_mapping)

def parse_prediction(prediction):
    if prediction == 0:
        return "NO"
    else:
        return "YES"

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

def summary(model):
    # summary of the model
    print('model intercept :', model.intercept_)
    print('model coefficients : ', model.coef_)

if(os.path.exists("RandomForest.joblib")):
    #Using our trained model to make predictions
    model = "RandomForest.joblib"
    RFC_Model = joblib.load(model)

    #spotify login
    load_dotenv()
    #save your credentials in a .env file using the following format
        #CLIENT_ID=your_client_id
        #CLIENT_SECRET=your_client_secret
    #to replicate the results.
    CLIENT_ID = os.getenv("CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
    OUTPUT_FILE_NAME = "track_info.csv"

    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )
    session = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    #USER INTERACTION: enter a song uri
    print("PLEASE, paste here only the part in the uri between 'track:' and '?'.... For example something like 6AgCJm3qeHplWeL2NFWh8w")
    song_name = input("Enter a song uri on spotify to predict if it will be popular: ")
    track = session.track(song_name)
    popularity = track["popularity"]
    song = session.audio_features('spotify:track:' + song_name)
   
    #info about the song
    print("Song name: " + track["name"])
    print(song)

    #creating a dataframe with the song infgo
    song_info = pd.DataFrame(song)
    print(song_info.head() )
    song_info.drop(['analysis_url','id','loudness','track_href','type','uri'], axis=1, inplace=True)
    #ordering the columns
    song_info = song_info[["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", 
                "mode", "speechiness", "tempo", "time_signature", "valence"]]

    #predicting the popularity
    prediction = RFC_Model.predict(song_info)
    print(popularity)
    print("The song will be popular: " + parse_prediction(prediction))




else:
    correlation = dataframe[['popularity','danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','duration_ms']].corr()
    
    print(correlation)

    list_of_keys = dataframe['key'].unique()
    for i in range(len(list_of_keys)):
        dataframe.loc[dataframe['key'] == list_of_keys[i], 'key'] = i
    print(dataframe.sample(5))



    #making popularity binary
    dataframe.loc[dataframe['popularity'] < 57, 'popularity'] = 0 
    dataframe.loc[dataframe['popularity'] >= 57, 'popularity'] = 1

    print(dataframe['popularity'])

    features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness", 
                "mode", "speechiness", "tempo", "time_signature", "valence"]

    training = dataframe.sample(frac = 0.8,random_state = 420)
    X_train = training[features]
    y_train = training['popularity']
    X_test = dataframe.drop(training.index)[features]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 420)



    print("LOGISTIC REGRESSION")
    print("--------------------------------------------------")
    LR_Model = LogisticRegression()
    LR_Model.fit(X_train, y_train)
    LR_Predict = LR_Model.predict(X_valid)
    LR_Accuracy = accuracy_score(y_valid, LR_Predict)
    print("Accuracy: " + str(LR_Accuracy))

    LR_AUC = roc_auc_score(y_valid, LR_Predict) 
    print("AUC: " + str(LR_AUC))

    print("--------------------------------------------------")
    print("RANDOM FOREST")
    print("--------------------------------------------------")
    RFC_Model = RandomForestClassifier()
    RFC_Model.fit(X_train, y_train)
    RFC_Predict = RFC_Model.predict(X_valid)
    RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
    print("Accuracy: " + str(RFC_Accuracy))
    filename = "RandomForest.joblib"
    joblib.dump(RFC_Model, filename)
    RFC_AUC = roc_auc_score(y_valid, RFC_Predict) 
    print("AUC: " + str(RFC_AUC))


    print("--------------------------------------------------")
    print("KNN")
    print("--------------------------------------------------")
    KNN_Model = KNeighborsClassifier()
    KNN_Model.fit(X_train, y_train)
    KNN_Predict = KNN_Model.predict(X_valid)
    KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
    print("Accuracy: " + str(KNN_Accuracy))

    KNN_AUC = roc_auc_score(y_valid, KNN_Predict) 
    print("AUC: " + str(KNN_AUC))

    print("--------------------------------------------------")
    print("DECISION TREE")
    print("--------------------------------------------------")
    DT_Model = DecisionTreeClassifier()
    DT_Model.fit(X_train, y_train)
    DT_Predict = DT_Model.predict(X_valid)
    DT_Accuracy = accuracy_score(y_valid, DT_Predict)
    print("Accuracy: " + str(DT_Accuracy))

    DT_AUC = roc_auc_score(y_valid, DT_Predict) 
    print("AUC: " + str(DT_AUC))

    training_LSVC = training.sample(10000)
    X_train_LSVC = training_LSVC[features]
    y_train_LSVC = training_LSVC['popularity']
    X_test_LSVC = dataframe.drop(training_LSVC.index)[features]
    X_train_LSVC, X_valid_LSVC, y_train_LSVC, y_valid_LSVC = train_test_split(
        X_train_LSVC, y_train_LSVC, test_size = 0.2, random_state = 420)

    print("--------------------------------------------------")
    print("LINEAR SVC")
    print("--------------------------------------------------")
    LSVC_Model = DecisionTreeClassifier()
    LSVC_Model.fit(X_train_LSVC, y_train_LSVC)
    LSVC_Predict = LSVC_Model.predict(X_valid_LSVC)
    LSVC_Accuracy = accuracy_score(y_valid_LSVC, LSVC_Predict)
    print("Accuracy: " + str(LSVC_Accuracy))

    LSVC_AUC = roc_auc_score(y_valid_LSVC, LSVC_Predict) 
    print("AUC: " + str(LSVC_AUC))

    print("--------------------------------------------------")
    print("XGBOOST")
    print("--------------------------------------------------")
    XGB_Model = XGBClassifier(objective = "binary:logistic", n_estimators = 10, seed = 123)
    XGB_Model.fit(X_train, y_train)
    XGB_Predict = XGB_Model.predict(X_valid)
    XGB_Accuracy = accuracy_score(y_valid, XGB_Predict)
    print("Accuracy: " + str(XGB_Accuracy))

    XGB_AUC = roc_auc_score(y_valid, XGB_Predict) 
    print("AUC: " + str(XGB_AUC))
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("MODEL PERFORMANCE")
    model_performance_accuracy = pd.DataFrame({'Model': ['LogisticRegression', 
                                                        'RandomForestClassifier', 
                                                        'KNeighborsClassifier',
                                                        'DecisionTreeClassifier',
                                                        'LinearSVC',
                                                        'XGBClassifier'],
                                                'Accuracy': [LR_Accuracy,
                                                            RFC_Accuracy,
                                                            KNN_Accuracy,
                                                            DT_Accuracy,
                                                            LSVC_Accuracy,
                                                            XGB_Accuracy]})

    model_performance_AUC = pd.DataFrame({'Model': ['LogisticRegression', 
                                                        'RandomForestClassifier', 
                                                        'KNeighborsClassifier',
                                                        'DecisionTreeClassifier',
                                                        'LinearSVC',
                                                        'XGBClassifier'],
                                                'AUC': [LR_AUC,
                                                            RFC_AUC,
                                                            KNN_AUC,
                                                            DT_AUC,
                                                            LSVC_AUC,
                                                            XGB_AUC]})


    print(model_performance_accuracy.sort_values(by = "Accuracy", ascending = False))

    plt.figure(figsize=(10, 6))
    plt.bar(model_performance_accuracy['Model'], model_performance_accuracy['Accuracy'], color='blue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance - Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Display the plot
    plt.show()



    print(model_performance_AUC.sort_values(by = "AUC", ascending = False))

    plt.figure(figsize=(10, 6))
    plt.bar(model_performance_AUC['Model'], model_performance_AUC['AUC'], color='green')
    plt.xlabel('Model')
    plt.ylabel('AUC')
    plt.title('Model Performance - AUC')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Display the plot
    plt.show()