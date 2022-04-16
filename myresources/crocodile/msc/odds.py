
from crocodile.file_management import P, install_n_import
from crocodile.meta import Scheduler, Log
import datetime
import time
import sys
import numpy as np


def qr(txt): install_n_import("qrcode").make(txt).save((file := P.tmpfile(suffix=".png")).__str__()); return file()
def count_number_of_lines_of_code_in_repo(path=P.cwd(), extension=".py", r=True, **kwargs): return P(path).search(f"*{extension}", r=r, **kwargs).read_text(encoding="utf-8").splitlines().apply(len).np.sum()


class Cycle:
    def __init__(self, c=None, name=''):
        self.c = c  # a list of values.
        self.index = -1
        self.name = name

    def next(self):
        self.index += 1
        if self.index >= len(self.c): self.index = 0
        return self.c[self.index]

    def previous(self):
        self.index -= 1
        if self.index < 0: self.index = len(self.c) - 1
        return self.c[self.index]

    def set(self, value): self.index = self.c.index(value)
    def get(self): return self.c[self.index]
    def get_index(self): return self.index
    def set_index(self, index): self.index = index
    def sample(self, size=1): return np.random.choice(self.c, size)
    def __add__(self, other): pass  # see behviour of matplotlib cyclers.
    def __str__(self): return self.name


class DictCycle(Cycle):
    def __init__(self, strct, **kwargs):
        strct = dict(strct)
        super(DictCycle, self).__init__(c=strct.items(), **kwargs)
        self.keys = strct.keys()

    def set_key(self, key): self.index = list(self.keys).index(key)


def polygon_area(points):
    """Return the area of the polygon whose vertices are given by the sequence points.
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2)


def pomodoro(work=25, rest=5, repeats=4):
    logger = Log(name="pomodoro", file=False, stream=True)

    def loop():
        speak("Alright, time to start working...")
        start = datetime.datetime.now()
        while (diff := work - ((datetime.datetime.now() - start).seconds / 60)) > 0:
            logger.debug(f"Keep working. Time Left: {round(diff)} minutes, Time now: {datetime.datetime.now()}"); time.sleep(60 * 1)
        speak("Now, its time to take a break.")
        start = datetime.datetime.now()
        while (diff := rest - ((datetime.datetime.now() - start).seconds / 60)) > 0:
            logger.critical(f"Keep Resting. Time Left: {round(diff)} minutes, Time now: {datetime.datetime.now().time()}"); time.sleep(60 * 1)

    def speak(txt):
        install_n_import("gtts").gTTS(txt, lang='en', tld='com.au').save(tmp := P.tmpfile(suffix=".mp3")); time.sleep(0.5)
        pyglet = install_n_import("pyglet"); pyglet.resource.path = [tmp.parent.str]; pyglet.resource.reindex(); pyglet.resource.media(tmp.name).play()

    def beep(duration=1, frequency=3000):
        try: import winsound
        except ImportError:
            __import__("os").system('beep -f %s -l %s' % (frequency, 1000 * duration))
        else: winsound.Beep(frequency, 1000 * duration)

    return Scheduler(routine=loop, max_cycles=repeats, logger=logger, wait="0.1m").run()


def profile_memory(command):
    psutil = install_n_import("psutil")
    before = psutil.virtual_memory()
    exec(command)
    after = psutil.virtual_memory()
    print(f"Memory used = {(after.used - before.used) / 1e6}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pass
