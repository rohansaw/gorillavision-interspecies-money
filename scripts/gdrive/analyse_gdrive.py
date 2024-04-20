
    
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import pandas as pd
import tqdm

def main():
    # TODO how long does it take to just read the whole folder?
    # TODO throw an error message when token is not found or not valid anymore
    creds = Credentials.from_authorized_user_file('token.json')
    service = build('drive', 'v3', credentials=creds)

    folder_id = '1cmSBtc1ldj-AiLHyqLVlaAOdRVpG9tuL'  # Replace with your Google Drive folder ID
    # folder_id = '14Fw4Kr7vsMsYHm1hWMXIUWDYeSWVc-pn'
    # folder_id='1IrArSrwmJl47TIdc8UgcozMnmoqozKZS'
    items = read_contents(service, folder_id)
    print(len(items))
    pd.DataFrame(items).to_csv('files.csv', index=False)
    

def download_file(service, file_id, file_name, folder_path):
    """Download a file from Google Drive.

    Args:
        service: Authenticated Google Drive service instance.
        file_id (str): ID of the file to download.
        file_name (str): Name of the file to download.
        folder_path (str): Local path to the folder where the file will be saved.
    """
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    file_path = os.path.join(folder_path, file_name)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.close()

def read_contents(service, folder_id, subfolder=None):
    """Recursively download the contents of a Google Drive folder.

    Args:
        service: Authenticated Google Drive service instance.
        folder_id (str): ID of the Google Drive folder.
        local_folder_path (str): Local path to save the downloaded contents.
    """
    # Make sure the local folder exists
    # if not os.path.exists(local_folder_path):
        # os.makedirs(local_folder_path)

    is_first_level = True if subfolder is None else False
    # Query to list contents of the folder
    query = f"'{folder_id}' in parents"
    response = service.files().list(q=query, spaces='drive', supportsAllDrives=True, includeItemsFromAllDrives=True, fields='nextPageToken, files(id, name, mimeType,size)').execute()
    items = response.get('files', [])
    
    next_page_token = response.get('nextPageToken')

# If there are more pages of results, continue fetching them
    while next_page_token:
        response = service.files().list(q=query,
                                     spaces='drive',
                                     supportsAllDrives=True,
                                     includeItemsFromAllDrives=True,
                                     pageToken=next_page_token,
                                     fields='nextPageToken, files(id, name, mimeType, size)').execute()
        items.extend(response.get('files', []))
        next_page_token = response.get('nextPageToken')
    
    files = []
    
    
    iterator = tqdm.tqdm(items) if is_first_level else items

    for item in iterator:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Recurse into subfolders
            # new_folder_path = os.path.join(local_folder_path, item['name'])
            if subfolder is not None:
                new_subfolder  = os.path.join(subfolder, item['name'])
            else:
                new_subfolder = item['name']
            files.extend(read_contents(service, item['id'], subfolder=new_subfolder))
        else:
            if subfolder is not None:
                item['name'] = subfolder + '/'+ item['name']
                
            # Download files
            # print(subfolder + '/'+ item['name'])
            # print(int(item['size']) / 1024. / 1024.)
            
            # if int(item['size']) / 1024. / 1024. > 25:
                # print(name)
                

            files.append(item)
            # download_file(service, item['id'], item['name'], local_folder_path)
            

    return files

if __name__ == '__main__':
    main()
