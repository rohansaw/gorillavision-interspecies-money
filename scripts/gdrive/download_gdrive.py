# First get a list of all files cause than we can 

import pandas as pd

    
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import pandas as pd
from tqdm import tqdm

def main():
    df = pd.read_csv('files.csv')
    creds = Credentials.from_authorized_user_file('token.json')
    service = build('drive', 'v3', credentials=creds)

    # df = df.loc[df['name'].str.startswith('C16/')]
    
    local_folder_path = 'downloaded/'  # Replace with your desired local path
    
    df['name'] = df['name'].apply(lambda x: local_folder_path + x)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_id = row['id']
        file_name = row['name']
        
        if os.path.exists(file_name):
            continue
        
        folder_path = os.path.dirname(file_name)
        os.makedirs(folder_path, exist_ok=True)
        
        file_name = os.path.basename(file_name)
        
        
        download_file(service, file_id, file_name, folder_path)



def download_file(service, file_id, file_name, folder_path):
    """Download a file from Google Drive.

    Args:
        service: Authenticated Google Drive service instance.
        file_id (str): ID of the file to download.
        file_name (str): Name of the file to download.
        folder_path (str): Local path to the folder where the file will be saved.
    """
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True )
    file_path = os.path.join(folder_path, file_name)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.close()
    
if __name__ == '__main__':
    main()
