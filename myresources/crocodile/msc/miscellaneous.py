
from crocodile.file_management import P, List, install_n_import
import datetime
import time
import sys
import os
import numpy as np


def qr(txt): install_n_import("qrcode").make(txt).save((file := P.tmpfile(suffix=".png")).__str__()); return file()
def count_number_of_lines_of_code_in_repo(path=P.cwd(), extension=".py", r=True, **kwargs): return P(path).search(f"*{extension}", r=r, **kwargs).read_text(encoding="utf-8").splitlines().apply(len).np.sum()
def get_list_of_executables_defined_in_shell(): return List(os.environ["Path"].split(";")).apply(lambda x: P(x).search("*.exe")).reduce(lambda x, y: x+y).print()


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
    """Return the area of the polygon whose vertices are given by the
    sequence points.
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    # 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return abs(area / 2)


class Pomodoro:
    def __init__(self, work=25, break_=5):
        self.work = work  # minutes
        self.break_ = break_  # minutes

    @staticmethod
    def loop(minutes, label='work'):
        start = datetime.datetime.now()
        diff_sec = 0
        while diff_sec < minutes * 60:
            now = datetime.datetime.now()
            diff = (datetime.datetime.now() - start)
            diff_sec = diff.seconds
            time.sleep(2)
            print(f"{label} .... Time Left: {round(minutes - diff.seconds / 60)} minutes, "
                  f" Time now: {now.hour} : {now.minute}", end='\r')
        Pomodoro.beep(5)

    def run(self):
        # Work
        self.loop(minutes=self.work, label="Work")
        # start the break
        self.loop(minutes=self.break_, label="Break")
        # repeat
        self.run()

    @staticmethod
    def beep(duration=1, frequency=3000):
        duration = 1000 * duration
        try: import winsound
        except ImportError:
            import os # apt-get install beep
            os.system('beep -f %s -l %s' % (frequency, duration))
        else: winsound.Beep(frequency, duration)


def profile_memory(command):
    psutil = install_n_import("psutil")
    before = psutil.virtual_memory()
    exec(command)
    after = psutil.virtual_memory()
    print(f"Memory used = {(after.used - before.used) / 1e6}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pass
