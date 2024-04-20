from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
import io
import os

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

def download_folder_contents(service, folder_id, local_folder_path):
    """Recursively download the contents of a Google Drive folder.

    Args:
        service: Authenticated Google Drive service instance.
        folder_id (str): ID of the Google Drive folder.
        local_folder_path (str): Local path to save the downloaded contents.
    """
    # Make sure the local folder exists
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)

    # Query to list contents of the folder This does not return all files and foldersgit
    query = f"'{folder_id}' in parents"
    response = service.files().list(q=query, spaces='drive',                                  supportsAllDrives=True,
                                 includeItemsFromAllDrives=True, fields='files(id, name, mimeType)').execute()
    items = response.get('files', [])

    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Recurse into subfolders
            new_folder_path = os.path.join(local_folder_path, item['name'])
            download_folder_contents(service, item['id'], new_folder_path)
        else:
            # Download files
            print(item)
            download_file(service, item['id'], item['name'], local_folder_path)

def main():
    creds = Credentials.from_authorized_user_file('token.json')
    service = build('drive', 'v3', credentials=creds)

    folder_id = '1jzWcr0CTbIldkPCJTCdWwSeuP5yIPNbn'  # Replace with your Google Drive folder ID
    local_folder_path = 'test'  # Replace with your desired local path

    download_folder_contents(service, folder_id, local_folder_path)

if __name__ == '__main__':
    main()
