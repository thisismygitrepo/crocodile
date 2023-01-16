
# import os
import random
# import crocodile.toolbox as tb
import textwrap
import subprocess


class BoxStyles:
    language = ['ada-box', 'caml', 'boxquote', 'stone', 'tex-box', 'shell', 'simple' 'c', 'cc', 'html']
    scene = ['whirly', 'xes', 'columns', 'parchment', 'scroll', 'scroll-akn', 'diamonds', 'headline', 'nuke', 'spring', 'stark1']  # , 'important3'
    character = ['capgirl', 'cat', 'boy', 'girl', 'dog', 'mouse', 'santa', 'face', 'ian_jones', 'peek', 'unicornsay']


COW_EYES = ['-b', '-d', '-g', '-h', '-l', '-L', '-n', '-N', '-p', '-s', '-t', '-w', '-y']
COW_FIGURES = ['apt', 'bunny', 'cheese', 'cock', 'cower', 'daemon', 'default', 'dragon',
               'dragon-and-cow', 'duck', 'elephant', 'elephant-in-snake', 'eyes', 'fox', 'ghostbusters',
               'gnu', 'kangaroo', 'kiss', 'milk',
               'moose', 'pony', 'pony-smaller', 'sheep', 'skeleton', 'snowman', 'stegosaurus',  # 'suse',
               'three-eyes', 'turkey', 'turtle', 'tux', 'unipony', 'unipony-smaller', 'vader', 'vader']  # 'hellokitty' 'mech-and-cow'  # 'moofasa', 'stimpy', 'calvin', , 'ren', 'koala', 'flaming-sheep' , 'bud-frogs' , 'kosh' , 'luke-koala'

FIGLET_FONTS = ['banner', 'big', 'standard']


def get_art(comment=None, artlib=None, style=None, super_style='scene', calliagraphy=False, prefix='', file=None, verbose=True):
    if comment is None: comment = subprocess.run("fortune", shell=True, capture_output=True, text=True).stdout
    if artlib is None: artlib = random.choice(['boxes', r'/usr/games/cowsay'])
    if calliagraphy is None: calliagraphy = True if artlib == 'boxes' else False
    if style is None:
        pool = {f'/usr/games/cowsay': COW_FIGURES, 'boxes': BoxStyles.__dict__[super_style or random.choice(['language', 'scene', 'character'])]}[artlib]
        style = random.choice(pool)
    # res = tb.Terminal().run(f"""echo "{comment}" | {boxlib} {'-d' if boxlib == 'boxes' else '-f'} {style} """).op
    cmd = f"""echo "{comment}" | {f'figlet -f {random.choice(FIGLET_FONTS)} | ' if calliagraphy else ''} {artlib} {'-d' if artlib == 'boxes' else '-f'} {style} {'' if not file else f'> {file}'}"""
    # print(cmd)
    res = subprocess.run(cmd, text=True, capture_output=True, shell=True).stdout
    res = textwrap.indent(res, prefix=prefix)
    if verbose:
        print(f'Using style: {style} from {artlib}', '\n' * 3)
        print(res)
    return res


if __name__ == '__main__':
    pass
