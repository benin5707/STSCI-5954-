import os #to access env variables and ensure functionality regardless of OS
from dotenv import load_dotenv
import lyricsgenius
import json
from requests.exceptions import HTTPError, Timeout, ConnectionError

#gets the genius access token for API querying
load_dotenv()

#For reference - getenv is hidden by gitignore for the code reviewer
TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

#compile the list of songs to query
failed_songs = [] #apparently faster lookup
with open("Data/failed_lyrics.jsonl", mode="r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        failed_songs.append([record["track_name"],record["artist_name"], record["track_id"]])

#create Genius class to do querying
lyricWriter = lyricsgenius.Genius(TOKEN, #copy pasted documentation
                                response_format='plain', #might write to json might write to parquet
                                timeout=10, 
                                sleep_time=3, #was 5, set to 3 in the interest of time
                                #verbose=False, #don't need to print everything
                                #apparently verbose was removed from the latest versions
                                remove_section_headers=True, #we don't need these for lyric analysis 
                                skip_non_songs=True, 
                                excluded_terms=None, 
                                replace_default_terms=False, 
                                retries=1, #set to 3 retries, altered to 1 in the interest of time
                                #user_agent='', unsure about usage
                                proxy=None)
#file to write to
with (
    open("Data/geniusLyrics.jsonl", mode = "a", encoding = "utf-8") as file,
    open("Data/failedGenius.jsonl", mode = "a", encoding = "utf-8") as FailF,
    ):
    #loop to query
    #for song in failed_songs:#comment [:1] to test with 1 track
    for i in range(2509,len(failed_songs)): #adjust the range to when it terminates
        #for some reason starting the range at 971 got the 971
        print(i)
        song = failed_songs[i]
        try:
            if not isinstance(song[1], str) or not song[1].strip(): #bypassing the 4 invalid entries
                   continue
            result = lyricWriter.search_song(song[0], song[1])
            #Check that it actually got something and not empty lyrics
            if result is not None and result.lyrics.strip():
                #create a dictionary to write to jsonl
                record = {
                    "track_id": song[2],
                    "track_name": song[0],
                    "track_artist": song[1],
                    "lyrics": result.lyrics
                    }
                #write to the json file and add new line
                file.write(json.dumps(record) + "\n")
            else:
                record = {
                    "track_id": song[2],
                    "track_name": song[0],
                    "track_artist": song[1],
                    "Error_type": "song not found"
                    }
                #write
                FailF.write(json.dumps(record) + "\n")
        except HTTPError as e: #taken from the lyricsgeniusdocumentation
            record = {#rewritten to not print error messages but rather store them with 
                #track id and relevant info
                "track_id": song[2],
                "track_name": song[0],
                "track_artist": song[1],
                "Error_type": "HTTPError",
                "Error_Status": e.args[0],
                "Error_message": e.args[1]
                }
            #write this error to the fail files
            FailF.write(json.dumps(record) + "\n")
        except Timeout:
            record = {
                "track_id": song[2],
                "track_name": song[0],
                "track_artist": song[1],
                "Error_type": "Timeout",
                }
            #write it now
            FailF.write(json.dumps(record) + "\n")
        except ConnectionError as e:
            record = {#rewritten to not print error messages but rather store them with 
                #track id and relevant info
                "track_id": song[2],
                "track_name": song[0],
                "track_artist": song[1],
                "Error_type": "Connection Error",
                }
            #write it
            FailF.write(json.dumps(record) + "\n")
        except Exception as e: #for unexpected stuff
            record = {#rewritten to not print error messages but rather store them with 
                #track id and relevant info
                "track_id": song[2],
                "track_name": song[0],
                "track_artist": song[1],
                "Error_type": "Unexpected",
                "Error_Status": e.args[0],
                "Error_message": str(e) #altered after 971 which caused an error
                }
            #write this error to the fail files
            FailF.write(json.dumps(record) + "\n")
            
        
            
        
        
