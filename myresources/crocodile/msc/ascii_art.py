
"""Ascii art
"""

import os
import random
#
import textwrap
import subprocess
from crocodile.core import install_n_import, randstr
# from pathlib import Path
from crocodile.file_management import P
import platform
from typing import Optional, Literal

# https://github.com/VaasuDevanS/cowsay-python
# https://github.com/pwaller/pyfiglet
# https://github.com/tehmaze/lolcat/
# https://github.com/wasi-master/gradient_figlet/
# https://github.com/sepandhaghighi/art


BOX_OR_CHAR = Literal['boxes', 'cowsay']


class ArtLib:
    @staticmethod
    def cowsay(text: str):
        cowsay = install_n_import("cowsay")
        char = random.choice(cowsay.char_names)
        return cowsay.get_output_string(char, text=text)
    # @staticmethod
    # def figlet(text: str):
    #     pyfiglet = install_n_import("pyfiglet")
    #     fs = pyfiglet.FigletFont.getFonts()
    #     f = pyfiglet.Figlet(font=random.choice(fs))
    #     return f.renderText(text)


class BoxStyles:
    language = ['ada-box', 'caml', 'boxquote', 'stone', 'tex-box', 'shell', 'simple', 'c', 'cc', 'html']
    scene = ['whirly', 'xes', 'columns', 'parchment', 'scroll', 'scroll-akn', 'diamonds', 'headline', 'nuke', 'spring', 'stark1']  # , 'important3'
    character = ['capgirl', 'cat', 'boy', 'girl', 'dog', 'mouse', 'santa', 'face', 'ian_jones', 'peek', 'unicornsay']


class CowStyles:
    eyes = ['-b', '-d', '-g', '-h', '-l', '-L', '-n', '-N', '-p', '-s', '-t', '-w', '-y']
    # this one for the package installed with sudo apt install cowsay and is located at /usr/games/cowsay. See cowsay -l
    figures = ['apt', 'bunny', 'cheese', 'cock', 'cower', 'daemon', 'default', 'dragon',
               'dragon-and-cow', 'duck', 'elephant', 'elephant-in-snake', 'eyes', 'fox', 'ghostbusters',
               'gnu', 'kangaroo', 'kiss', 'milk',
               'moose', 'pony', 'pony-smaller', 'sheep', 'skeleton', 'snowman', 'stegosaurus',  # 'suse',
               'three-eyes', 'turkey', 'turtle', 'tux', 'unipony', 'unipony-smaller', 'vader', 'vader']  # 'hellokitty' 'mech-and-cow'  # 'moofasa', 'stimpy', 'calvin', , 'ren', 'koala', 'flaming-sheep' , 'bud-frogs' , 'kosh' , 'luke-koala'


FIGLET_FONTS = ['banner', 'big', 'standard']

FIGJS_FONTS = ['3D Diagonal', '3D-ASCII', '4Max', '5 Line Oblique', 'Acrobatic', 'ANSI Regular', 'ANSI Shadow',
               'Avatar', 'Banner', 'Banner3-D', 'Banner4',
               'Basic', 'Big Money-ne', 'Big Money-nw', 'Big Money-se', 'Big Money-sw', 'Big', 'Bloody', 'Bolger', 'Braced', 'Bright',
               'DOS Rebel',
               'Elite', 'Epic', 'Flower Power',
               'Fraktur',  # 'Isometric4'. 'AMC Tubes', 'Banner3', Alligator2
               'Star Wars',
               'Sub-Zero', 'The Edge', 'USA Flag', 'Varsity', "Doom"
               ]  # too large  Crazy 'Sweet', 'Electronic', 'Swamp Land', Crawford, Alligator


def get_art(comment: Optional[str] = None, artlib: Optional[BOX_OR_CHAR] = None, style: Optional[str] = None, super_style: str = 'scene', prefix: str = ' ', file: Optional[str] = None, verbose: bool = True):
    """ takes in a comment and does the following wrangling:
    * text => figlet font => boxes => lolcat
    * text => cowsay => lolcat
    """
    if comment is None: comment = subprocess.run("fortune", shell=True, capture_output=True, text=True, check=True).stdout
    if artlib is None: artlib = random.choice(['boxes', 'cowsay'])
    to_file = '' if not file else f'> {file}'
    if artlib == 'boxes':
        if style is None: style = random.choice(BoxStyles.__dict__[super_style or random.choice(['language', 'scene', 'character'])])
        fonting = f'figlet -f {random.choice(FIGLET_FONTS)}'
        cmd = f"""echo "{comment}" | {fonting} | boxes -d {style} {to_file}"""
        # tmpfile = Path.home().joinpath("tmp_results/tmp_figlet.txt")
        # tmpfile.write_text(ArtLib.figlet(comment))
        # cmd = f"""cat {tmpfile} | boxes -d {style} {to_file}"""
    else:
        if style is None: style = random.choice(CowStyles.figures)
        cmd = f"""echo "{comment}" | cowsay -f {style} {to_file}"""
    try:
        res = subprocess.run(cmd, text=True, capture_output=True, shell=True, check=True).stdout
    except subprocess.CalledProcessError as ex:
        print(ex)
        return ""
    res = textwrap.indent(res, prefix=prefix)
    if verbose:
        print(f'Using style: {style} from {artlib}', '\n' * 3)
        print(f'{cmd=}')
        print('Results:\n', res)
    return res


def font_box_color(logo: str):
    # from crocodile.msc.ascii_art import FIGJS_FONTS  # , BoxStyles # pylint: disable=C0412
    font = random.choice(FIGJS_FONTS)
    # print(f"{font}\n")
    box_style = random.choice(['whirly', 'xes', 'columns', 'parchment', 'scroll', 'scroll-akn', 'diamonds', 'headline', 'nuke', 'spring', 'stark1'])
    _cmd = f'figlet -f "{font}" "{logo}" | boxes -d "{box_style}" | lolcatjs'
    # print(_cmd)
    os.system(_cmd)  # | lolcat
    # print("after")


def character_color(logo: str):
    assert platform.system() == 'Windows', 'This function is only for Windows.'
    # from rgbprint import gradient_scroll, Color
    # gradient_scroll(ArtLib.cowsay("crocodile"), start_color=0x4BBEE3, end_color=Color.medium_violet_red, times=3)
    _new_art = P.tmp().joinpath("tmp_arts").create().joinpath(f"{randstr()}.txt")
    _new_art.write_text(ArtLib.cowsay(logo))  # utf-8 encoding?
    os.system(f'type {_new_art} | lolcatjs')  # | lolcat


def character_or_box_color(logo: str):
    assert platform.system() == 'Linux', 'This function is only for Linux.'
    _new_art = P.tmp().joinpath("tmp_arts").create().joinpath(f"{randstr()}.txt")
    get_art(logo, artlib=None, file=str(_new_art), verbose=False)
    command = f"cat {_new_art} | lolcat"
    os.system(command)  # full path since lolcat might not be in PATH.
    # try:
    #     output = subprocess.check_output(_cmd, shell=True, stderr=subprocess.STDOUT)
    # except subprocess.CalledProcessError as e:
    #     print(f"Command '{_cmd}' returned non-zero exit status {e.returncode}. Output: {e.output}")


if __name__ == '__main__':
    pass
