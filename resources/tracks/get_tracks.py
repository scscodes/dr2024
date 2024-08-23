import os
import requests
from dotenv import load_dotenv


load_dotenv()
github_token = os.getenv('GITHUB_TOKEN')

# GitHub API URL to list contents of the 'npy' folder in the repository
url = "https://api.github.com/repos/aws-deepracer-community/deepracer-race-data/contents/raw_data/tracks/npy"

# Set up the request headers with the GitHub token
headers = {
    "Authorization": f"token {github_token}"
}


def download_file(file_name, download_url):
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        response = requests.get(download_url)
        if response.status_code == 200:
            # Write the content to a local file
            with open(file_name, 'wb') as file:
                file.write(response.content)
        else:
            print(f".. Failed {file_name}: {response.status_code}")
    else:
        print(f".. Skipping (exists): {file_name}.")


def download_files_from_github(url, headers):
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            file_name = file['name']
            download_url = file['download_url']
            download_file(file_name, download_url)
    else:
        print(f"Failed to retrieve files: {response.status_code}")


download_files_from_github(url, headers)
