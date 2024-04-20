from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def list_files_in_folder(folder_id):
    """List all files and directories within a specific Google Drive folder.

    Args:
        folder_id (str): The ID of the folder to list contents from.

    Returns:
        list: A list of dictionaries, each representing a file or folder.
    """
    creds = None
    # Load your credentials from the file. Replace 'token.json' with the path to your credentials file.
    creds = Credentials.from_authorized_user_file('token.json')

    service = build('drive', 'v3', credentials=creds)

    results = []
    # Query to search for files and folders within the specified folder.
    query = f"'{folder_id}' in parents"
    # Call the Drive v3 API to list the contents of the folder.
    items = service.files().list(q=query,
                                 supportsAllDrives=True,
                                 includeItemsFromAllDrives=True,
                                 spaces='drive',
                                 fields='nextPageToken, files(id, name, mimeType)').execute().get('files', [])

    for item in items:
        # For each item, store its id, name, and type (file or folder).
        results.append({
            'id': item['id'],
            'name': item['name'],
            'type': 'folder' if item['mimeType'] == 'application/vnd.google-apps.folder' else 'file'
        })

    return results

# Example usage:
# Replace 'YOUR_FOLDER_ID_HERE' with the actual folder ID.
folder_id = '1l9LuXED86muI3XVwgc7_aiwoVUNGJh7F'
contents = list_files_in_folder(folder_id)
for content in contents:
    print(f"Name: {content['name']}, Type: {content['type']}")
