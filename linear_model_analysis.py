import csv
import os
import re

import spotipy
import pandas as pd
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns; 
sns.set()


# load credentials from .env file
load_dotenv()
#save your credentials in a .env file using the following format
    #CLIENT_ID=your_client_id
    #CLIENT_SECRET=your_client_secret
#to replicate the results.
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
OUTPUT_FILE_NAME = "track_info.csv"

# target playlist
PLAYLIST_LINK = "https://open.spotify.com/playlist/37i9dQZF1DWSxF6XNtQ9Rg?si=efc4b6ea5d79470a"

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
    writer.writerow(["track", "artist","track_id","popularity"])

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

#create a new dataframe with audio features
features_list = []
for features in audio_features:
    features_list.append([features[0]['danceability'], features[0]['energy'], features[0]['loudness'], features[0]['speechiness'], features[0]['acousticness'], features[0]['instrumentalness'], features[0]['liveness'], features[0]['valence'], features[0]['tempo'], features[0]['duration_ms']])
df_audio_features = pd.DataFrame(features_list, columns=['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'])
#save the dataframe to a csv file
df_audio_features.to_csv('audio_features.csv', index=False)


#extracting a model to predict popularity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_squared_error, r2_score
#creating the model based on audio features
X_d = df_audio_features[['danceability']]
X_e = df_audio_features[['energy']]
X_l = df_audio_features[['loudness']]
X_s = df_audio_features[['speechiness']]
X_a = df_audio_features[['acousticness']]
X_i = df_audio_features[['instrumentalness']]
X_li = df_audio_features[['liveness']]
X_v = df_audio_features[['valence']]
X_t = df_audio_features[['tempo']]
X_du = df_audio_features[['duration_ms']]
y = df['popularity']

#metto a rapporto le features con la popolarità
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y, test_size=0.2, random_state=0)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_e, y, test_size=0.2, random_state=0)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y, test_size=0.2, random_state=0)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y, test_size=0.2, random_state=0)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y, test_size=0.2, random_state=0)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_i, y, test_size=0.2, random_state=0)
X_train_li, X_test_li, y_train_li, y_test_li = train_test_split(X_li, y, test_size=0.2, random_state=0)
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_v, y, test_size=0.2, random_state=0)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y, test_size=0.2, random_state=0)
X_train_du, X_test_du, y_train_du, y_test_du = train_test_split(X_du, y, test_size=0.2, random_state=0)


#creo un modello generale con tutte le features per fare le previsioni
X = df_audio_features[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#uso il modello di regressione di Poisson
y_train = list(y_train)
model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
#stampo il summary
print(model.summary())

#faccio le previsioni
y_pred = model.predict(X_test)
#stampo il r2 score
print(r2_score(y_test, y_pred))
'''
#plotting the results
plt.scatter(y_test, y_pred, color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.show()
'''
#testo il modello con le features più significative
X = df_audio_features[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = list(y_train)
model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
print(model.summary())
y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))

#testo con le features più significative e tolgo le variabili che hanno un p-value alto
X = df_audio_features[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence']]
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = list(y_train)
model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
print(model.summary())
y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))

#provo a fare una previsione reale con una canzone
#prendo i dati della canzone
track = session.track("6YNuS3tJfip0Xqw9Ixzint")
popularity = track["popularity"]
song = session.audio_features('spotify:track:71ZrOdx1qBU48QKt0Eppbn')
#creo un dataframe con i dati della canzone
df_song = pd.DataFrame(song)
#creo un dataframe con le features più significative
df_song_features = df_song[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence']]
#stampo il dataframe
print(df_song_features)
#faccio la previsione
prediction = model.predict(df_song_features)
#stampo la previsione
print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
#stampo la popolarità della canzone
print(popularity)

print("MODELLO GENERALE")
#provo a trainare il modello con il dataset più grande
#carico il dataset
df = pd.read_csv('SpotifyAudioFeaturesNov2018.csv')
#elimino le colonne che non mi servono
df = df.drop(['artist_name', 'track_id', 'track_name', 'key', 'mode', 'time_signature'], axis=1)
#elimino le righe con valori nulli
df = df.dropna()
#creo il modello
X = df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = list(y_train)
model_biggerData = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
print(model_biggerData.summary())

#provo a fare una previsione reale con una canzone
#prendo i dati della canzone
track = session.track("6YNuS3tJfip0Xqw9Ixzint")
popularity = track["popularity"]
song = session.audio_features('spotify:track:71ZrOdx1qBU48QKt0Eppbn')
#creo un dataframe con i dati della canzone
df_song = pd.DataFrame(song)
#creo un dataframe con le features più significative
df_song_features = df_song[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
#stampo il dataframe
print(df_song_features)
#faccio la previsione
prediction = model_biggerData.predict(df_song_features)
#stampo la previsione
print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
#stampo la popolarità della canzone
print(popularity)

#provo a fare una previsione reale con una canzone
#prendo i dati della canzone
track = session.track("2TtZg4cx4LdFv398URrl5x")
popularity = track["popularity"]
song = session.audio_features('spotify:track:2TtZg4cx4LdFv398URrl5x')
#creo un dataframe con i dati della canzone
df_song = pd.DataFrame(song)
#creo un dataframe con le features più significative
df_song_features = df_song[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
#stampo il dataframe
print(df_song_features)
#faccio la previsione
prediction = model_biggerData.predict(df_song_features)
#stampo la previsione
print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
#stampo la popolarità della canzone
print(popularity)

#provo a fare una previsione reale con una canzone
#prendo i dati della canzone
track = session.track("6AgCJm3qeHplWeL2NFWh8w")
popularity = track["popularity"]
song = session.audio_features('spotify:track:6AgCJm3qeHplWeL2NFWh8w')
#creo un dataframe con i dati della canzone
df_song = pd.DataFrame(song)
#creo un dataframe con le features più significative
df_song_features = df_song[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
#stampo il dataframe
print(df_song_features)
#faccio la previsione
prediction = model_biggerData.predict(df_song_features)
#stampo la previsione
print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
#stampo la popolarità della canzone
print(popularity)





#CAMBIO ALGORITMO
#Generalized Linear Mixed Effects Model GLIMMIX
#trattiamo la popolarità come un fixed effect e le altre features come random effect
print("MODELLO CON GLIMMIX")
#carico il dataset
df = pd.read_csv('SpotifyAudioFeaturesNov2018.csv')
#elimino le colonne che non mi servono
df = df.drop(['track_id', 'track_name', 'key', 'mode', 'time_signature'], axis=1)
#elimino le righe con valori nulli
df = df.dropna()
md = smf.mixedlm("popularity ~ danceability", df, groups=df["artist_name"])
model_mixed = md.fit()
print(model_mixed.summary())

performance = pd.DataFrame()
performance['artist_name'] = df['artist_name']
performance['popularity'] = df['popularity']
performance['danceability'] = df['danceability']
performance['energy'] = df['energy']
performance['loudness'] = df['loudness']
performance['speechiness'] = df['speechiness']
performance['valence'] = df['valence']
performance['predicted_popularity'] = model_mixed.predict()
plt.scatter(performance['popularity'], performance['predicted_popularity'])
plt.xlabel('popularity')
plt.ylabel('predicted_popularity')
plt.show()

#provo a fare una previsione reale con una canzone
#provo a fare una previsione reale con una canzone
#prendo i dati della canzone
track = session.track("6YNuS3tJfip0Xqw9Ixzint")
popularity = track["popularity"]
song = session.audio_features('spotify:track:71ZrOdx1qBU48QKt0Eppbn')
#creo un dataframe con i dati della canzone
df_song = pd.DataFrame(song)
#creo un dataframe con le features più significative
df_song_features = df_song[['danceability', 'energy', 'loudness', 'speechiness', 'valence']]
#stampo il dataframe
print(df_song_features)
#faccio la previsione
prediction = model_mixed.predict(df_song_features)
#stampo la previsione
print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
#stampo la popolarità della canzone
print(popularity)

#provo a fare una previsione reale con una canzone
#prendo i dati della canzone
track = session.track("2TtZg4cx4LdFv398URrl5x")
popularity = track["popularity"]
song = session.audio_features('spotify:track:2TtZg4cx4LdFv398URrl5x')
#creo un dataframe con i dati della canzone
df_song = pd.DataFrame(song)
#creo un dataframe con le features più significative
df_song_features = df_song[['danceability', 'energy', 'loudness', 'speechiness', 'valence']]
#stampo il dataframe
print(df_song_features)
#faccio la previsione
prediction = model_mixed.predict(df_song_features)
#stampo la previsione
print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
#stampo la popolarità della canzone
print(popularity)

#provo a fare una previsione reale con una canzone
#prendo i dati della canzone
track = session.track("6AgCJm3qeHplWeL2NFWh8w")
popularity = track["popularity"]
song = session.audio_features('spotify:track:6AgCJm3qeHplWeL2NFWh8w')
#creo un dataframe con i dati della canzone
df_song = pd.DataFrame(song)
#creo un dataframe con le features più significative
df_song_features = df_song[['danceability', 'energy', 'loudness', 'speechiness', 'valence']]
#stampo il dataframe
print(df_song_features)
#faccio la previsione
prediction =  model_mixed.predict(df_song_features)
#stampo la previsione
print("PREDICTED POPULARITY: "+prediction.to_string(index=False))
#stampo la popolarità della canzone
print(popularity)