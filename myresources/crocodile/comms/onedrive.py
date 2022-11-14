
"""
References:
    * https://learn.microsoft.com/en-us/graph/tutorials/python?tabs=aad
    * https://github.com/pranabdas/Access-OneDrive-via-Microsoft-Graph-Python

"""

import requests
import json
import urllib
# import os
from getpass import getpass
import time
from datetime import datetime
import crocodile.toolbox as tb


URL_AUTH = 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize'
URL_TOKEN = 'https://login.microsoftonline.com/common/oauth2/v2.0/token'
URL_GRAPH = 'https://graph.microsoft.com/v1.0/'
redirect_uri = 'http://localhost:8080/'


class OneDrive:
    def __init__(self, account=None):
        self.creds_path = tb.P.home().joinpath("dotfiles/microsoft/onedrive/config.toml")
        self.creds_content = self.creds_path.readit()
        self.account = account
        creds = self.creds_content[self.creds_content['accounts']['default']] if account is None else self.creds_content[account]
        self.creds = creds
        self.client_id = creds['client_id']
        self.client_secret = creds['client_secret']
        self.permissions = ['offline_access', 'files.readwrite', 'User.Read']

        self.token, self.refresh_token, self.last_updated = creds['token'], creds['refresh_token'], creds['last_updated']
        self.headers = {'Authorization': 'Bearer ' + self.token}

        scope = ''
        for items in range(len(self.permissions)):
            scope = scope + self.permissions[items]
            if items < len(self.permissions)-1:
                scope = scope + '+'
        self.scope = scope
        self.get_refresh_token()

    def save_creds(self):
        creds = self.creds
        creds['token'] = self.token
        creds['refresh_token'] = self.refresh_token
        creds['last_updated'] = self.last_updated
        self.creds_path.writeit(creds)

    def token_flow_auth(self):
        response_type = 'token'
        print('Click over this link ' + URL_AUTH + '?client_id=' + self.client_id + '&scope=' + self.scope + '&response_type=' + response_type + '&redirect_uri=' + urllib.parse.quote(redirect_uri))
        print('Sign in to your account, copy the whole redirected URL.')
        code = input("Paste the URL here :")
        token = code[(code.find('access_token') + len('access_token') + 1): (code.find('&token_type'))]
        headers = {'Authorization': 'Bearer ' + token}
        response = requests.get(URL_GRAPH + 'me/drive/', headers=headers)
        print(json.loads(response.text))
        if response.status_code == 200:
            response = json.loads(response.text)
            print('Connected to the OneDrive of', response['owner']['user']['displayName']+' (', response['driveType']+' ).', '\nConnection valid for one hour. Reauthenticate if required.')
        elif response.status_code == 401:
            response = json.loads(response.text)
            print('API Error! : ', response['error']['code'], '\nSee response for more details.')
        else:
            response = json.loads(response.text)
            print('Unknown error! See response for more details.')
            print(response)

    def code_flow_auth(self):
        # returns access_token and refresh_token  for persistent session.
        response_type = 'code'
        print('Click over this link ' + URL_AUTH + '?client_id=' + self.client_id + '&scope=' + self.scope + '&response_type=' + response_type + '&redirect_uri=' + urllib.parse.quote(redirect_uri))
        print('Sign in to your account, copy the whole redirected URL.')
        code = getpass("Paste the URL here :")
        code = code[(code.find('?code') + len('?code') + 1):]
        response = requests.post(URL_TOKEN + '?client_id=' + self.client_id + '&scope=' + self.scope + '&grant_type=authorization_code' + '&redirect_uri=' + urllib.parse.quote(redirect_uri) + '&code=' + code)
        _ = response
        data = {
            "client_id": self.client_id,
            "scope": self.permissions,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": 'authorization_code',
            "client_secret": self.client_secret
        }

        response = requests.post(URL_TOKEN, data=data)
        try:
            token = json.loads(response.text)["access_token"]
            refresh_token = json.loads(response.text)["refresh_token"]
        except KeyError:
            print(response.text)
            raise KeyError
        last_updated = time.mktime(datetime.today().timetuple())
        self.token, self.refresh_token, self.last_updated = token, refresh_token, last_updated

    def get_refresh_token(self):
        data = {
            "client_id": self.client_id,
            "scope": self.permissions,
            "refresh_token": self.refresh_token,
            "redirect_uri": redirect_uri,
            "grant_type": 'refresh_token',
            "client_secret": self.client_secret,
        }
        response = requests.post(URL_TOKEN, data=data)
        try:
            token = json.loads(response.text)["access_token"]
            refresh_token = json.loads(response.text)["refresh_token"]
        except KeyError:
            print(response.text)
            raise KeyError
        last_updated = time.mktime(datetime.today().timetuple())
        self.token, self.refresh_token, self.last_updated = token, refresh_token, last_updated
        self.headers = {'Authorization': 'Bearer ' + self.token}

    def connect(self):
        response = requests.get(URL_GRAPH + 'me/drive/', headers=self.headers)

        if response.status_code == 200:
            response = json.loads(response.text)
            print('Connected to the OneDrive of', response['owner']['user']['displayName']+' (', response['driveType']+' ).', '\nConnection valid for one hour. Refresh token if required.')
        elif response.status_code == 401:
            response = json.loads(response.text)
            print('API Error! : ', response['error']['code'], '\nSee response for more details.')
            print(response)
        else:
            response = json.loads(response.text)
            print('Unknown error! See response for more details.')

    def list_items(self):
        items = json.loads(requests.get(URL_GRAPH + 'me/drive/root/children', headers=self.headers).text)
        items = items['value']


def main():
    pass


if __name__ == '__main__':
    d = OneDrive()
