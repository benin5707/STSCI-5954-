import json

failed_ids = set() #apparently faster lookup
with open("Data/failed_lyrics.jsonl", mode="r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        failed_ids.add(record["track_id"])

#just storing the track ids instead of a dictionary now
#originally stored as dictionary

import csv
matchingRows = []
with open("Data/spotify_songs.csv", mode = "r", encoding="utf-8") as compareFile:
    csvDict = csv.DictReader(compareFile) #convert it to a dictionary
    for song in csvDict:
        if song["track_id"] in failed_ids:
            matchingRows.append(song)

#Checking output
#print(matchingRows[0])

#parameters I need to keep to query genius 
paraList = ["track_name", "track_artist"]
with (
    open("notWordy.csv", mode = "w", encoding="utf-8", newline="") as discard,
    open("extraSongs.csv", mode = "w", encoding="utf-8", newline="") as keep,
    ):
    KeepWrite = csv.DictWriter(keep, fieldnames= paraList, extrasaction= "ignore")
    DiscardWrite = csv.DictWriter(discard, fieldnames= paraList, extrasaction= "ignore")
    #Note apparently need writeheader to actually create the row in csv despite fieldnames
    KeepWrite.writeheader()
    DiscardWrite.writeheader()

    for track in matchingRows:
        #.33 threshold is from the tidytuesday readme 
        if float(track['speechiness']) < .33:
            DiscardWrite.writerow(track)
        else:
            KeepWrite.writerow(track)
    
    


