
from crocodile.file_management import P, install_n_import, datetime
from crocodile.meta import Scheduler, Log
import time


def share(path):
    response = install_n_import("requests").post(url=r'https://file.io', data=dict(name=path.name, expires="2022-08-01", title="a descriptivefile title", maxDownloads=1, autoDelete=True, private=False,
                                                 description="its a file, init?", ), files={'file': path.read_bytes()})
    return response.json()['link'] if ['link'] in response.json() else response.json()
def edit_video(path, t_start=0, t_end=None, speed=1, suffix=None, rotate=0, volume=1.0, fps=25):
    from moviepy.editor import VideoFileClip, vfx
    clip = VideoFileClip(path); print(f"{clip.size=}, {clip.duration=}, {clip.fps=}")
    clip.subclip(t_start=t_start, t_end=t_end).rotate(rotate).volumex(volume).fx(vfx.speedx, speed).write_videofile(path.append("_modified").with_suffix(path.suffix if suffix is None else suffix), fps=fps)


def qr(txt): install_n_import("qrcode").make(txt).save((file := P.tmpfile(suffix=".png")).__str__()); return file()
def capture_photo(): cv2 = install_n_import("cv2", "opencv-python"); cam = cv2.VideoCapture(); cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3000); cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000); _, frame = cam.read(); cam.releaes(); return frame
def count_number_of_lines_of_code_in_repo(path=P.cwd(), extension=".py", r=True, **kwargs): return P(path).search(f"*{extension}", r=r, **kwargs).read_text(encoding="utf-8").splitlines().apply(len).np.sum()
def profile_memory(command): psutil = install_n_import("psutil"); before = psutil.virtual_memory(); exec(command); after = psutil.virtual_memory(); print(f"Memory used = {(after.used - before.used) / 1e6}")
def pomodoro(work=25, rest=5, repeats=4):
    logger = Log(name="pomodoro", file=False, stream=True)
    def loop(sched):
        speak("Alright, time to start working..."); start = datetime.now(); _ = sched
        while (diff := work - ((datetime.now() - start).seconds / 60)) > 0: logger.debug(f"Keep working. Time Left: {round(diff)} minutes"); time.sleep(60 * 1)
        speak("Now, its time to take a break."); start = datetime.now()
        while (diff := rest - ((datetime.now() - start).seconds / 60)) > 0: logger.critical(f"Keep Resting. Time Left: {round(diff)} minutes"); time.sleep(60 * 1)
    def speak(txt):
        install_n_import("gtts").gTTS(txt, lang='en', tld='com.au').save(tmp := P.tmpfile(suffix=".mp3")); time.sleep(0.5)
        pyglet = install_n_import("pyglet"); pyglet.resource.path = [tmp.parent.str]; pyglet.resource.reindex(); pyglet.resource.media(tmp.name).play()
    def beep(duration=1, frequency=3000):
        try: import winsound
        except ImportError: __import__("os").system('beep -f %s -l %s' % (frequency, 1000 * duration))
        else: winsound.Beep(frequency, 1000 * duration)
    return Scheduler(routine=loop, max_cycles=repeats, logger=logger, wait="0.1m").run()


class Cycle:
    def __init__(self, iterable=None): self.list = iterable; self.index = -1
    def next(self): self.index += 1; self.index = 0 if self.index >= len(self.list) else self.index; return self.list[self.index]
    def previous(self): self.index -= 1; self.index = len(self.list) - 1 if self.index < 0 else self.index; return self.list[self.index]
    def set(self, value): self.index = self.list.index(value)
    def get(self): return self.list[self.index]
    def get_index(self): return self.index
    def set_index(self, index): self.index = index
    def __add__(self, other): pass  # see behviour of matplotlib cyclers.
class DictCycle(Cycle):
    def __init__(self, strct): super(DictCycle, self).__init__(iterable=strct.items()); self.keys = strct.keys()
    def set_key(self, key): self.index = list(self.keys).index(key)


def capture_from_webcam(show=True):
    cv2 = install_n_import("cv2", "opencv-python")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if show: cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows(); return frame


def tree(self, level: int = -1, limit_to_directories: bool = False, length_limit: int = 1000, stats=False, desc=None):
    # Based on: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python"""
    space, branch, tee, last, dir_path, files, directories = '    ', '│   ', '├── ', '└── ', self, 0, 0
    def get_stats(apath): return (f" {(sts := apath.stats(printit=False)).size} MB. {sts.content_mod_time}. " + desc(apath) if desc is not None else "") if stats or desc else ""
    def inner(apath: P, prefix: str = '', level_=-1):
        nonlocal files, directories
        if not level_: return  # 0, stop iterating
        pointers = [tee] * (len(content := apath.search("*", files=not limit_to_directories)) - 1) + [last]
        for pointer, path in zip(pointers, content):
            if path.is_dir():
                yield prefix + pointer + path.name + get_stats(path)
                directories, extension = directories + 1, branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level_=level_ - 1)
            elif not limit_to_directories: yield prefix + pointer + path.name + get_stats(path); files += 1
    print(dir_path.name); iterator = inner(dir_path, level_=level)
    [print(line) for line in __import__("itertools").islice(iterator, length_limit)]; print(f'... length_limit, {length_limit}, reached, counted:') if next(iterator, None) else None; print(f'\n{directories} directories' + (f', {files} files' if files else ''))


if __name__ == '__main__':
    pass
