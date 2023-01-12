
"""
This module uses the generic google-api-python-client, which is very low level.

read: https://developers.google.com/drive/api/guides/search-files
file specs: https://developers.google.com/drive/api/v3/reference/files
# https://github.com/eduardogr/google-drive-python
# google: python gdrive api expiry date

# code from https://developers.google.com/drive/api/quickstart/python
terminology: https://developers.google.com/workspace/guides/auth-overview

Steps:
* Create a project on console.developers.google.com which redirects to https://console.cloud.google.com/projectselector2/apis/dashboard
* Enable the Drive API
* Create credentials for a web server to access application data
"""

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from crocodile.comms.helper_funcs import process_sent_file, process_retrieved_file
from enum import Enum
import crocodile.toolbox as tb
import io
import pandas as pd


tb.Display.set_pandas_display()


class Scopes(Enum):  # TODO: If modifying these scopes, delete the file token.json.
    admin =      tb.P('https://www.googleapis.com/auth/drive')  # See, edit, create, and delete all of your Google Drive files
    settings =   tb.P('https://www.googleapis.com/auth/drive.appdata')  # View and manage its own configuration data in your Google Drive
    manage =     tb.P('https://www.googleapis.com/auth/drive.file')  # View and manage Google Drive files and folders that you have opened or created with this app
    managemeta = tb.P('https://www.googleapis.com/auth/drive.metadata')  # View and manage metadata of files in your Google # Drive
    readmeta =   tb.P('https://www.googleapis.com/auth/drive.metadata.readonly')  # View metadata for files in your Google Drive
    photo =      tb.P('https://www.googleapis.com/auth/drive.photos.readonly')  # View the photos, videos and albums in your # Google Photos
    readonly =   tb.P('https://www.googleapis.com/auth/drive.readonly')  # See and download all your Google Drive files
    app =        tb.P('https://www.googleapis.com/auth/drive.scripts')  # Modify your Google Apps Script scripts' behavior


class GDriveAPI:
    def __init__(self, account=None, project=None, scopes=None):
        scopes = [Scopes.admin] if scopes is None else scopes
        self.SCOPES = [scope.value.as_url_str() for scope in scopes]
        self.creds = self.get_cred(account, project)
        self.service = build('drive', 'v3', credentials=self.creds)
        self.drive_url = tb.P("https://drive.google.com/drive/u/1")

    def get_cred(self, account, project):
        config = tb.P.home().joinpath("dotfiles/google/drive/config.toml").readit()
        if account is None: account = list(config.keys())[0]; print(f"GDRIVE: using default account `{account}`")
        if project is None: project = list(config[account].keys())[0]; print(f"GDRIVE: using default project `{project}`")
        api_key = config[account][project]['api_key']
        client_id_file = tb.P.home().joinpath(f"dotfiles/google/drive/{config[account][project]['auth_client_config']}")
        creds = None  # The file token.json stores the user's access and refresh tokens, and is created automatically when the authorization flow completes for the first time.

        if client_id_file.exists():
            # client_id_info = client_id_file.readit()
            # if "refresh_token" in client_id_info:  # refersh token is not available before at least the first use.
            creds = Credentials.from_authorized_user_file(filename=client_id_file, scopes=self.SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:  # If there are no (valid) credentials available, let the user log in.
            if creds and creds.expired and creds.refresh_token:
                try: creds.refresh(Request())
                except Exception: creds = self.authorize_screen()
            else: creds = self.authorize_screen()
            client_id_file.write_text(creds.to_json())
        return creds

    def authorize_screen(self):
        auth_secret = tb.P.home().joinpath(f"dotfiles/google/drive/Oauth_client_secret.json")
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file=auth_secret, scopes=self.SCOPES)
        creds = flow.run_local_server(port=0)
        return creds

    def search_all(self, q='', size=1000): return pd.DataFrame(self.service.files().list(q=q, pageSize=size, fields="nextPageToken, files(id, name)", spaces="drive").execute().get('files', []), columns=['id', 'name']).set_index("name")
    def search_folders(self): return pd.DataFrame(self.service.files().list(q="mimeType = 'application/vnd.google-apps.folder'").execute()['files'])
    def get_fields_from_id(self, fid, fields="*") -> dict: return self.service.files().get(fileId=fid, fields=fields).execute()  # e.g: fields="id, name, mimeType, size, modifiedTime, createdTime, trashed, parents"
    def get_path_from_id(self, fid) -> str:
        tmp = self.get_fields_from_id(fid, 'name, parents')
        name = tmp['name']
        parent_id = tmp['parents'][0] if 'parents' in tmp else None
        if parent_id is None: return name  # name will be 'My Drive'
        else: return self.get_path_from_id(parent_id) + "/" + name

    def get_id_from_path(self, path) -> str:
        parent_id = 'root'
        for item in tb.P(path).items:
            tmp = self.service.files().list(q=f"(name={repr(item)}) and ({repr(parent_id)} in parents)", fields="nextPageToken, files(kind, id, name, parents)").execute()['files']
            assert len(tmp) < 2, f"Mutiple files with same name `{item}`. Couldn't resolve ambiguity."
            assert len(tmp) > 0, FileNotFoundError(f"FileNotFoundError `{item}`.")
            parent_id = tmp[0]['id']
        return parent_id
    def update(self, remote_path=None, fid=None, local_path=None):
        fid = fid or self.get_id_from_path(remote_path)
        print(f"UPDATING contents of file `{fid}`")
        return self.service.files().update(fileId=fid, media_body=MediaFileUpload(local_path)).execute()

    @staticmethod
    def get_link_from_id(fid, folder=False):
        return tb.P(rf'https://drive.google.com/file/d/{fid}') if not folder else tb.P(rf'https://drive.google.com/drive/u/1/folders/{fid}')
    @staticmethod
    def get_id_from_link(link): return str(link).split(r"https://drive.google.com/file/d/")[1]

    def download(self, fpath=None, fid=None, furl=None, local_dir=None, rel2home=False, decrypt=False, unzip=False, key=None, pwd=None):
        assert fpath or fid or furl, "Either a file name or a file id must be provided."
        if furl is not None: fid = self.get_id_from_link(furl)
        if fpath is not None:
            if rel2home:
                fpath = tb.P(fpath).expanduser().absolute()
                local_dir = fpath.parent
                fpath = tb.get_env().myhome / fpath.rel2home()
            if unzip: fpath = (fpath + ".zip") if ".zip" not in str(fpath) else fpath
            if decrypt: fpath = (fpath + ".enc") if ".enc" not in str(fpath) else fpath
            fid = self.get_id_from_path(fpath)
        else: fpath = self.get_path_from_id(fid)
        local_dir = tb.P(local_dir or f'~/Downloads').expanduser().create()
        fields = self.get_fields_from_id(fid, fields="name, mimeType")
        if fields['mimeType'] == 'application/vnd.google-apps.folder':
            return tb.L(self.service.files().list(q=f"'{fid}' in parents").execute()['files']).apply(lambda x: self.download(fid=x['id'], local_dir=local_dir.joinpath(fields['name'])))
        downloader = MediaIoBaseDownload(fh := io.BytesIO(), self.service.files().get_media(fileId=fid))
        while True:
            status, done = downloader.next_chunk()  # fh: file lives in the RAM now.
            print(f"DOWNLOADING from GDrive {fpath} ==> {local_dir}. {int(status.progress() * 100)}%.")
            if done: break
        fh.seek(0);
        path = local_dir.joinpath(fields['name']).write_bytes(fh.read())
        process_retrieved_file(path, decrypt=decrypt, unzip=unzip, key=key, pwd=pwd)
        return path

    def upload(self, local_path, remote_dir="", overwrite=True, share=False, rel2home=False, zip_first=False, encrypt_first=False, key=None, pwd=None):
        local_path = process_sent_file(file=local_path, zip_first=zip_first, encrypt_first=encrypt_first, key=key, pwd=pwd)
        if rel2home: remote_dir = tb.get_env().myhome.joinpath(local_path.rel2home().parent)
        else: remote_dir = tb.P(remote_dir)
        try: self.get_id_from_path(remote_dir)
        except AssertionError as ae:
            if "FileNotFoundError" in str(ae): self.create_folder(remote_dir)
            else: raise NotImplementedError(f"{ae}")
        print(f"UPLOADING {repr(local_path)} to {repr(remote_dir)} ... ", end="")
        file = {'id': None}
        if local_path.is_dir():
            self.create_folder(path=remote_dir.joinpath(local_path.name))
            for item in local_path.listdir(): self.upload(local_path.joinpath(item), remote_dir=remote_dir.joinpath(local_path.name), zip_first=zip_first, encrypt_first=encrypt_first, key=key, pwd=pwd)
        else:  # upload a file.
            try:
                existing_fid = self.get_id_from_path(remote_dir.joinpath(local_path.name))
                if overwrite: file = self.update(local_path=local_path, fid=existing_fid)
                else: raise FileExistsError(f"File `{local_path.name}` already exists in `{remote_dir}`.")
            except AssertionError as ae:
                if "FileNotFoundError" in str(ae): pass  # good, it's a new file.
                elif "Couldn't resolve ambiguity." in str(ae): raise NotImplementedError("Please manually delete files with same name. " + str(ae))
                file_metadata = {'name': local_path.name, 'parents': [self.get_id_from_path(remote_dir)], 'role': 'reader', 'type': 'anyone'}
                file = self.service.files().create(body=file_metadata, media_body=MediaFileUpload(local_path.str), fields='id').execute()
                print(f"file id: {file.get('id')}")
        if zip_first or encrypt_first: tb.P(local_path).delete(sure=True)
        return {"remote_path": remote_dir.joinpath(local_path.name), 'local_path': local_path, 'fid': file['id']}

    def upload_and_share(self, local_path, overwrite=True, role='writer', zip_first=False, encrypt_first=False, key=None, pwd=None, rel2home=False):
        # https://developers.google.com/drive/api/guides/manage-sharing
        res = self.upload(local_path=local_path, remote_dir="myshare", overwrite=overwrite, share=True, zip_first=zip_first, encrypt_first=encrypt_first, key=key, pwd=pwd, rel2home=rel2home)
        self.service.permissions().create(fileId=res["fid"], body=dict(type='anyone', role=role)).execute()
        res['url'] = tb.P(rf'https://drive.google.com/file/d/{res["fid"]}')
        return res

    def create_folder(self, path="") -> str:
        path = tb.P(path)
        parent_id = 'root'
        for part in path.parts:
            try:
                search_res = self.service.files().list(q=f"'{parent_id}' in parents and name = '{part}' and mimeType = 'application/vnd.google-apps.folder'").execute()['files']
                parent_id = search_res[0]['id']
            except Exception as e:  # not found ==> create it
                file_metadata = {
                    'name': part,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [parent_id]}
                file = self.service.files().create(body=file_metadata, fields='id').execute()
                print(f"CREATED FOLDER `{part}` in `{path}`. ID = {file.get('id')}")
                parent_id = file.get('id')
        return parent_id


"""
Tip: folders and files on G drive have unique ID that is not the name. Thus, there can be multiple files with same name. 
This can cause seemingly confusing behaviour if the user continues to rely on names rather than IDs. 
Folders in trash bin are particularly confusing as the trash bin is nothing but another folder and thus,
user can still access it and populate it, etc. This action taking place in rubbish bin might not be noticed when inspecting Gdrive via web.
"""


if __name__ == '__main__':
    # pass
    api = GDriveAPI()
