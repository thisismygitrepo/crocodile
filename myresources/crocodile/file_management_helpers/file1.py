
from pathlib import Path
from typing import Optional, Union, Callable


# %% =============================== Security ================================================
def obscure(msg: bytes) -> bytes:
    import base64
    import zlib
    return base64.urlsafe_b64encode(zlib.compress(msg, 9))
def unobscure(obscured: bytes) -> bytes:
    import zlib
    import base64
    return zlib.decompress(base64.urlsafe_b64decode(obscured))
def hashpwd(password: str):
    import bcrypt
    return bcrypt.hashpw(password=password.encode(), salt=bcrypt.gensalt()).decode()
def pwd2key(password: str, salt: Optional[bytes] = None, iterations: int = 10) -> bytes:  # Derive a secret key from a given password and salt"""
    import base64
    if salt is None:
        import hashlib
        m = hashlib.sha256()
        m.update(password.encode(encoding="utf-8"))
        return base64.urlsafe_b64encode(s=m.digest())  # make url-safe bytes required by Ferent.
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    return base64.urlsafe_b64encode(PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=None).derive(password.encode()))
def encrypt(msg: bytes, key: Optional[bytes] = None, pwd: Optional[str] = None, salted: bool = True, iteration: Optional[int] = None, gen_key: bool = False) -> bytes:
    import base64
    from cryptography.fernet import Fernet
    salt, iteration = None, None
    if __debug__:
        from crocodile.file_management_helpers.file3 import P
        _ = P
    if pwd is not None:  # generate it from password
        assert (key is None) and (type(pwd) is str), "❌ You can either pass key or pwd, or none of them, but not both."
        import secrets
        iteration = iteration or secrets.randbelow(exclusive_upper_bound=1_000_000)
        salt = secrets.token_bytes(nbytes=16) if salted else None
        key_resolved = pwd2key(password=pwd, salt=salt, iterations=iteration)
    elif key is None:
        if gen_key:
            key_resolved = Fernet.generate_key()
            Path.home().joinpath('dotfiles/creds/data/encrypted_files_key.bytes').write_bytes(key_resolved)
        else:
            try:
                key_resolved = Path.home().joinpath("dotfiles/creds/data/encrypted_files_key.bytes").read_bytes()
                print(f"⚠️ Using key from: {Path.home().joinpath('dotfiles/creds/data/encrypted_files_key.bytes')}")
            except FileNotFoundError as err:
                print("\n" * 3, "~" * 50, """Consider Loading up your dotfiles or pass `gen_key=True` to make and save one.""", "~" * 50, "\n" * 3)
                raise FileNotFoundError(err) from err
    elif isinstance(key, (str, P, Path)): key_resolved = Path(key).read_bytes()  # a path to a key file was passed, read it:
    elif type(key) is bytes: key_resolved = key  # key passed explicitly
    else: raise TypeError("❌ Key must be either a path, bytes object or None.")
    code = Fernet(key=key_resolved).encrypt(msg)
    if pwd is not None and salt is not None and iteration is not None: return base64.urlsafe_b64encode(b'%b%b%b' % (salt, iteration.to_bytes(4, 'big'), base64.urlsafe_b64decode(code)))
    return code
def decrypt(token: bytes, key: Optional[bytes] = None, pwd: Optional[str] = None, salted: bool = True) -> bytes:
    import base64
    if __debug__:
        from crocodile.file_management_helpers.file3 import P
        _ = P
    if pwd is not None:
        assert key is None, "❌ You can either pass key or pwd, or none of them, but not both."
        if salted:
            decoded = base64.urlsafe_b64decode(token)
            salt, iterations, token = decoded[:16], decoded[16:20], base64.urlsafe_b64encode(decoded[20:])
            key_resolved = pwd2key(password=pwd, salt=salt, iterations=int.from_bytes(bytes=iterations, byteorder='big'))
        else: key_resolved = pwd2key(password=pwd)  # trailing `;` prevents IPython from caching the result.
    elif type(key) is bytes:
        assert pwd is None, "❌ You can either pass key or pwd, or none of them, but not both."
        key_resolved = key  # passsed explicitly
    elif key is None: key_resolved = Path.home().joinpath("dotfiles/creds/data/encrypted_files_key.bytes").read_bytes()  # read from file
    elif isinstance(key, (str, P, Path)): key_resolved = Path(key).read_bytes()  # passed a path to a file containing kwy
    else: raise TypeError(f"❌ Key must be either str, P, Path, bytes or None. Recieved: {type(key)}")
    from cryptography.fernet import Fernet
    return Fernet(key=key_resolved).decrypt(token)
def unlock(drive: str = "D:", pwd: Optional[str] = None, auto_unlock: bool = False):
    from crocodile.meta import Terminal
    s1 = f"""$SecureString = ConvertTo-SecureString "{pwd or Path.home().joinpath("dotfiles/creds/data/bitlocker_pwd").read_text(encoding="utf-8")}" -AsPlainText -Force; Unlock-BitLocker -MountPoint "{drive}" -Password $SecureString; """
    return Terminal().run(s1 + (f'Enable-BitLockerAutoUnlock -MountPoint "{drive}"' if auto_unlock else ''), shell="powershell").print(desc="Unlocking Bitlocker Drive")


def modify_text(txt_raw: str, txt_search: str, txt_alt: Union[str, Callable[[str], str]], replace_line: bool = True, notfound_append: bool = False, prepend: bool = False, strict: bool = False):
    lines, bingo = txt_raw.split("\n"), False
    if not replace_line:  # no need for line splitting
        assert isinstance(txt_alt, str), f"txt_alt must be a string if notfound_append is True. It is not: {txt_alt}"
        if txt_search in txt_raw: return txt_raw.replace(txt_search, txt_alt)
        return txt_raw + "\n" + txt_alt if notfound_append else txt_raw
    for idx, line in enumerate(lines):
        if txt_search in line:
            if isinstance(txt_alt, str): lines[idx] = txt_alt
            elif callable(txt_alt): lines[idx] = txt_alt(line)
            bingo = True
    if strict and not bingo: raise ValueError(f"txt_search `{txt_search}` not found in txt_raw `{txt_raw}`")
    if bingo is False and notfound_append is True:
        assert isinstance(txt_alt, str), f"txt_alt must be a string if notfound_append is True. It is not: {txt_alt}"
        if prepend: lines.insert(0, txt_alt)
        else: lines.append(txt_alt)  # txt not found, add it anyway.
    return "\n".join(lines)
