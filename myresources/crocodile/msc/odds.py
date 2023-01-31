
from crocodile.file_management import P, install_n_import, datetime
from crocodile.core import List
from crocodile.meta import Scheduler, Log
import time


def share(path):
    response = install_n_import("requests").post(url=r'https://file.io', data=dict(name=path.name, expires="2022-08-01", title="a descriptivefile title", maxDownloads=1, autoDelete=True, private=False, description="its a file, init?", ), files={'file': path.read_bytes()})
    return response.json()['link'] if ['link'] in response.json() else response.json()
def edit_video(path, t_start=0, t_end=None, speed=1, suffix=None, rotate=0, volume=1.0, fps=25):
    ed = install_n_import("moviepy").editor
    clip = ed.VideoFileClip(path); print(f"{clip.size=}, {clip.duration=}, {clip.fps=}")
    clip.subclip(t_start=t_start, t_end=t_end).rotate(rotate).volumex(volume).fx(ed.vfx.speedx, speed).write_videofile(path.append("_modified").with_suffix(path.suffix if suffix is None else suffix), fps=fps)


def capture_from_webcam(show=True, wait=True, save=False):
    cv2 = install_n_import("cv2", "opencv-python")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3000)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)
    ret, frame = cam.read()
    while True and wait:
        ret, frame = cam.read()
        if show: cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cam.release()
    cv2.destroyAllWindows()
    if save:
        path = P.tmpfile(suffix=".jpg")
        cv2.imwrite(str(path), frame)
        return path
    return frame


def qr(txt): install_n_import("qrcode").make(txt).save((file := P.tmpfile(suffix=".png")).__str__()); return file()
def count_number_of_lines_of_code_in_repo(path=P.cwd(), extension=".py", r=True, **kwargs): return P(path).search(f"*{extension}", r=r, **kwargs).read_text(encoding="utf-8").splitlines().apply(len).np.sum()
def profile_memory(command): psutil = install_n_import("psutil"); before = psutil.virtual_memory(); exec(command); after = psutil.virtual_memory(); print(f"Memory used = {(after.used - before.used) / 1e6}")


class Cycle:
    def __init__(self, iterable=None, max_idx=None):
        if iterable is not None:
            self.list = iterable
            self.index, self.prev_index = 0, -1
        else:
            if max_idx is not None:
                self.list = list(range(max_idx))
                self.index, self.prev_index = 0, -1
            else:
                self.list = []
                self.index, self.prev_index = -1, -2
    def next(self): self.prev_index = self.index; self.index += 1; self.index = 0 if self.index >= len(self.list) else self.index; return self.list[self.index]
    def previous(self): self.prev_index = self.index; self.index -= 1; self.index = len(self.list) - 1 if self.index < 0 else self.index; return self.list[self.index]
    def set_value(self, value): self.prev_index = self.index; self.index = self.list.index(value)
    def get_value(self): return self.list[self.index]
    def get_index(self): return self.index
    def set_index(self, index): self.prev_index = self.index; self.index = index
    def expand(self, val=None): self.list += (val if type(val) is list else ([self.list[-1] + 1]) if len(self.list) else [0])
    def __add__(self, other): pass  # see behviour of matplotlib cyclers.
    def __repr__(self): return f"Cycler @ {self.index}: {self.list[self.index]}"
class DictCycle(Cycle):
    def __init__(self, strct): super(DictCycle, self).__init__(iterable=strct.items()); self.keys = strct.keys()
    def set_key(self, key): self.index = list(self.keys).index(key)


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
    print(dir_path.name)
    iterator = inner(dir_path, level_=level)
    [print(line) for line in __import__("itertools").islice(iterator, length_limit)]
    print(f'... length_limit, {length_limit}, reached, counted:') if next(iterator, None) else None
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))


def get_compressable_directories(path, max_size_mb=15_000):
    tmp_results = path.search("*", r=False)
    dirs2compress = List()
    dirs_violating = List()
    files_violating = List()
    for item in tmp_results:
        if item.size() > max_size_mb:  # should be parsed
            if item.is_file():
                print(f"File `{item}` has size larger than maximum allowed, consider manual handling")
                files_violating.append(item)
            else:
                tmp_dirs2compress, tmp_dirs_violating, tmp_files_violating = get_compressable_directories(item, max_size_mb=max_size_mb)
                if len(tmp_dirs2compress) == 0:
                    print(f"Directory `{item}` has size larger than maximum allowed, but when parsed for subdirectories to be compressed, nothing could be found. Handle manually")
                    dirs_violating.append(item)
                dirs2compress += tmp_dirs2compress
                dirs_violating += tmp_dirs_violating
                files_violating += tmp_files_violating
        else:
            if item.is_dir() and len(item.search("*", folders=False, r=True)) > 10:  # too many files ==> compress
                dirs2compress.append(item)
    return dirs2compress, dirs_violating, files_violating


if __name__ == '__main__':
    pass
