
"""Ascii art
"""

# import os
import random
# import crocodile.toolbox as tb
import textwrap
import subprocess
from crocodile.core import install_n_import
# from pathlib import Path
from typing import Optional

# https://github.com/VaasuDevanS/cowsay-python
# https://github.com/pwaller/pyfiglet
# https://github.com/tehmaze/lolcat/
# https://github.com/wasi-master/gradient_figlet/
# https://github.com/sepandhaghighi/art

"""

"""


class ArtLib:
    @staticmethod
    def cowsay(text: str):
        cowsay = install_n_import("cowsay")
        char = random.choice(cowsay.char_names)
        return cowsay.get_output_string(char_name=char, text=text)

    @staticmethod
    def figlet(text: str):
        pyfiglet = install_n_import("pyfiglet")
        fs = pyfiglet.FigletFont.getFonts()
        f = pyfiglet.Figlet(font=random.choice(fs))
        return f.renderText(text)


class BoxStyles:
    language = ['ada-box', 'caml', 'boxquote', 'stone', 'tex-box', 'shell', 'simple', 'c', 'cc', 'html']
    scene = ['whirly', 'xes', 'columns', 'parchment', 'scroll', 'scroll-akn', 'diamonds', 'headline', 'nuke', 'spring', 'stark1']  # , 'important3'
    character = ['capgirl', 'cat', 'boy', 'girl', 'dog', 'mouse', 'santa', 'face', 'ian_jones', 'peek', 'unicornsay']


class CowStyles:
    eyes = ['-b', '-d', '-g', '-h', '-l', '-L', '-n', '-N', '-p', '-s', '-t', '-w', '-y']
    figures = ['apt', 'bunny', 'cheese', 'cock', 'cower', 'daemon', 'default', 'dragon',
               'dragon-and-cow', 'duck', 'elephant', 'elephant-in-snake', 'eyes', 'fox', 'ghostbusters',
               'gnu', 'kangaroo', 'kiss', 'milk',
               'moose', 'pony', 'pony-smaller', 'sheep', 'skeleton', 'snowman', 'stegosaurus',  # 'suse',
               'three-eyes', 'turkey', 'turtle', 'tux', 'unipony', 'unipony-smaller', 'vader', 'vader']  # 'hellokitty' 'mech-and-cow'  # 'moofasa', 'stimpy', 'calvin', , 'ren', 'koala', 'flaming-sheep' , 'bud-frogs' , 'kosh' , 'luke-koala'


FIGLET_FONTS = ['banner', 'big', 'standard']

FIGJS_FONTS = ['3D Diagonal', '3D-ASCII', '4Max', '5 Line Oblique', 'Acrobatic', 'ANSI Regular', 'ANSI Shadow', 'Avatar', 'Banner', 'Banner3-D', 'Banner4',
               'Basic', 'Big Money-ne', 'Big Money-nw', 'Big Money-se', 'Big Money-sw', 'Big', 'Bloody', 'Bolger', 'Braced', 'Bright',
               'DOS Rebel',
               'Elite', 'Epic', 'Flower Power',
               'Fraktur',  # 'Isometric4'. 'AMC Tubes', 'Banner3', Alligator2
               'Star Wars',
               'Sub-Zero', 'The Edge', 'USA Flag', 'Varsity', "Doom"]  # too large  Crazy 'Sweet', 'Electronic', 'Swamp Land', Crawford, Alligator


def get_art(comment: Optional[str] = None, artlib: Optional[str] = None, style: Optional[str] = None, super_style: str = 'scene', prefix: str = ' ', file: Optional[str] = None, verbose: bool = True):
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
    res = subprocess.run(cmd, text=True, capture_output=True, shell=True, check=True).stdout
    res = textwrap.indent(res, prefix=prefix)
    if verbose:
        print(f'Using style: {style} from {artlib}', '\n' * 3)
        print(f'{cmd=}')
        print(f'Results:\n', res)
    return res


if __name__ == '__main__':
    pass
