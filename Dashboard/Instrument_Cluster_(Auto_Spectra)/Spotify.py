import socket
import spotipy
import json
from spotipy.oauth2 import SpotifyOAuth

# Spotify authentication
client_id = '3fb73f276ec048b78eff8151cee5563c'
client_secret = 'e8d229b5cc704e4d9c29bbd62957f93d'
redirect_uri = 'http://localhost:8888'
scope = "user-read-currently-playing"

sp = spotipy.SpotifyOAuth(client_id=client_id,
                          client_secret=client_secret,
                          redirect_uri=redirect_uri,
                          scope=scope)

token_dict = sp.get_access_token()
token = token_dict['access_token']

spotify_object = spotipy.Spotify(auth=token)

# UDP server address and port
udp_host = '127.0.0.1'  # Change to your UDP server's address
udp_port = 12356        # Change to your UDP server's port number

# Function to update the currently playing track information
def update_current_track():
    current_track = spotify_object.current_user_playing_track()
    try:
        if current_track is not None:
            json_data ={
                "trackName": current_track['item']['name'],
                "artistName": current_track['item']['artists'][0]['name'],
                "albumName": current_track['item']['album']['name'],
                "albumURL": current_track['item']['album']['images'][0]['url']
            }
            send_song_data(json_data)
            # print(json_data)
        else:
            json_data ={
                "trackName":"",
                "artistName":"",
                "albumName":"",
                "albumURL":""
            }
            send_song_data(json_data)
            # print(json_data)
    except:
            json_data ={
                "trackName": "Advertisement",
                "artistName": "Spotify",
                "albumName": "",
                "albumURL": "images/s.png"
            }
            send_song_data(json_data)
# Function to send image data over UDP
def send_song_data(json_data):
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    json_bytes = json.dumps(json_data).encode('utf-8')
    sock.sendto(json_bytes, (udp_host, udp_port))
    # Close the socket
    sock.close()

# Start updating the currently playing track information
while True:
    update_current_track()

