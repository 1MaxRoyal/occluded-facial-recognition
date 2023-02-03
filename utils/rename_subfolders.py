#renames subfolders a random word
import os
import requests

# folder path where the subfolders are located
folder_path = "../data/datasets/fei"

# function to get a random word
def get_random_word():
    response = requests.get("https://random-word-api.herokuapp.com/word?number=1")
    if response.status_code == 200:
        return response.json()[0]
    else:
        return "random_word"

# rename all subfolders
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        new_name = get_random_word()
        new_path = os.path.join(folder_path, new_name)
        os.rename(subfolder_path, new_path)
        print(f"Renamed {subfolder} to {new_name}")
