
import crocodile.toolbox as tb
import datetime
import time
import sys


def compute_num_of_lines_of_code_in_repo(path=tb.P.cwd(), extension=".py", r=True, **kwargs):
    return tb.P(path).search(f"*{extension}", r=r, **kwargs).read_text().splitlines().apply(len).np.sum()


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
        try:
            import winsound
        except ImportError:
            import os
            # apt-get install beep
            os.system('beep -f %s -l %s' % (frequency, duration))
        else:
            winsound.Beep(frequency, duration)


def commit_all_repos():
    tm = tb.Terminal()

    def commit_all(repos=None):
        if repos is None: repos = tb.P.home().joinpath("code")
        return repos.search("*").apply(lambda x: commit_one(x), verbose=True)

    def commit_one(path, mess="auto_commit_" + tb.randstr()):
        return tm.run(f'cd {path}; git add .; git commit -am "{mess}"; git push origin')

    return commit_all


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pass
