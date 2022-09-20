
import random
import crocodile.toolbox as tb
import textwrap


class Style:
    language = ['ada-box', 'boxquote', 'stone', 'tex-box', 'shell', 'simple' 'c', 'cc', 'html']
    scene = ['whirly', 'xes', 'columns', 'parchment', 'scroll', 'scroll-akn', 'diamonds', 'headline', 'nuke', 'spring', 'stark1']  # , 'important3'
    charachter = ['caml', 'capgirl', 'cat', 'boy', 'gril', 'dog', 'mouse', 'santa', 'face', 'ian_jones', 'peek', 'unicornsay']


def get_box(comment, prefix='', style=None, style_cat='scene'):
    if style is None: style = random.choice(Style.__dict__[style_cat])
    res = tb.Terminal().run(f"""echo "{comment}" | boxes -d {style} """)
    return textwrap.indent(res, prefix=prefix)


if __name__ == '__main__':
    pass
