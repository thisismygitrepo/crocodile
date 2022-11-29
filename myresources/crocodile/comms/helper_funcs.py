
import crocodile.toolbox as tb


def process_retrieved_file(path, decrypt=False, unzip=False, key=None, pwd=None):
    if decrypt: path = path.decrypt(key=key, pwd=pwd, inplace=True)
    if unzip: path = path.unzip(inplace=True, verbose=True, overwrite=True, content=True)
    return path


def process_sent_file(file, zip_first=False, encrypt_first=False, key=None, pwd=None):
    file = tb.P(file).expanduser().absolute()
    if zip_first: file = tb.P(file).zip()
    if encrypt_first:
        res = tb.P(file).encrypt(key=key, pwd=pwd)
        if zip_first: file.delete(sure=True)
        file = res
    return file


if __name__ == '__main__':
    pass
