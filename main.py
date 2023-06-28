import csv
import os
import re

import spotipy
import pandas as pd
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

import matplotlib.pyplot as plt
import sklearn

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
    features_list.append([features[0]['danceability'], features[0]['energy'], features[0]['loudness'], features[0]['speechiness'], features[0]['acousticness'], features[0]['instrumentalness'], features[0]['liveness'], features[0]['valence'], features[0]['tempo']])
df_audio_features = pd.DataFrame(features_list, columns=['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])
#save the dataframe to a csv file
df_audio_features.to_csv('audio_features.csv', index=False)



#extracting a model to predict popularity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#creating the model based on audio features
X = df_audio_features[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
print('Coefficients: \n', reg.coef_)
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
#plotting the results
plt.scatter(y_test, y_pred, color='black')
plt.plot(y_test, y_test, color='blue', linewidth=3)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.show()
#save the plot to a png file
plt.savefig('plot.png')

#predicting the popularity of a song
#example: 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
#example: 0.719, 0.704, -5.484, 0.0484, 0.00144, 0.000000, 0.0936, 0.562, 98.027
print(reg.predict([[0.719, 0.704, -5.484, 0.0484, 0.00144, 0.000000, 0.0936, 0.562, 98.027]]))
