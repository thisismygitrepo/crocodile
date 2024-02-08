
"""FM
"""


from crocodile.core import Base, randstr
from crocodile.file_management import P
from typing import Any, Optional
import base64


def obfuscate(seed: str, data: str):
    xored = ''.join(chr(ord(x) ^ ord(y)) for (x,y) in zip(data, seed*len(data)))
    return base64.urlsafe_b64encode(xored.encode()).decode()


def deobfuscate(seed: str, data: str):
    xored = base64.urlsafe_b64decode(data.encode()).decode()
    return ''.join(chr(ord(x) ^ ord(y)) for (x,y) in zip(xored, seed*len(xored)))


class Obfuscator(Base):
    def __init__(self, directory: str, noun: bool = False, suffix: str = "same") -> None:
        super().__init__()
        self.directory = P(directory).expanduser().absolute()
        assert self.directory.is_dir()
        self.real_to_phony: dict[str, str] = {}  # reltive string -> relative string
        self.phony_to_real: dict[str, str] = {}  # reltive string -> relative string
        self.noun = noun
        self.suffix = suffix

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['directory'] = self.directory.collapseuser(strict=False)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self.directory = self.directory.expanduser()

    def deobfuscate(self, path: P) -> P:
        directory_obf = self.obfuscate(self.directory)
        path_rel = directory_obf[-1] / path.relative_to(directory_obf)
        return self.directory.joinpath(self.phony_to_real[path_rel.as_posix()])

    def obfuscate(self, path: P) -> P:
        return self.directory.parent.joinpath(self.real_to_phony[path.relative_to(self.directory).as_posix()])

    def execute_map(self, forward: bool = True) -> None:
        root = self.directory if forward else self.obfuscate(self.directory)
        self._execute_map(root, forward=forward)

    def symlink_to_phoney(self, base: Optional[P] = None):
        base = P(base).expanduser().absolute() if base is not None else self.directory
        directory_obf = self.obfuscate(self.directory)
        for path_obf in directory_obf.search("*", r=True):
            if path_obf.is_dir(): continue
            path_real = self.deobfuscate(path_obf)
            base.joinpath(path_real.relative_to(self.directory)).symlink_to(path_obf)

    @staticmethod
    def lcd(path1: P, path2: P):
        it1 = P(path1).expanduser().absolute().parts
        it2 = P(path2).expanduser().absolute().parts
        parts = [part1 for part1, part2 in zip(it1, it2) if part1 == part2]
        return P("/".join(parts))

    def _execute_map(self, path: P, forward: bool) -> None:
        assert path.is_dir()
        children = path.search("*", r=False)
        for child in children:
            if child.is_file():
                if forward:
                    new_name = self.obfuscate(child).name
                else: new_name = self.deobfuscate(child).name
                child.with_name(new_name, inplace=True)
            else:
                self._execute_map(child, forward=forward)
        if forward: new_name = self.obfuscate(path).name
        else: new_name = self.deobfuscate(path).name
        path.with_name(new_name, inplace=True)

    def build_map(self):
        assert len(self.real_to_phony) == 0
        self._build_map(path=self.directory, path_r_obf="")
        self.display()

    def _build_map(self, path: P, path_r_obf: str):
        assert path.is_dir()
        children = path.search("*", r=False)
        path_r_obf = self._add_map(path, path_r_obf)
        for child in children:
            if child.is_file():
                self._add_map(child, path_r_obf)
            elif child.is_dir():
                self._build_map(child, path_r_obf)
            else: raise TypeError(f"Path is neither file nor directory. `{child}`")

    def _add_map(self, path: P, path_r_obf: str):
        if path.is_file():
            suffix = "".join(path.suffixes) if self.suffix == "same" else self.suffix
            file_name_obf = f"_0f_{randstr(noun=self.noun)}{suffix}"
            path_r_obf = path_r_obf + f"/{file_name_obf}"
        elif path.is_dir():
            if path_r_obf == "": path_r_obf = "__0d_root__" + randstr(noun=self.noun)
            else: path_r_obf = path_r_obf + f"/_0d_{randstr(noun=self.noun)}"
        k, v = path.relative_to(self.directory).as_posix(), path_r_obf
        self.real_to_phony[k] = v
        self.phony_to_real[v] = k
        return v

    def display(self):
        for k, v in self.real_to_phony.items():
            print(k)
            print(v)
            print('-' * 100)

    def save(self, **kwargs: Any):
        _ = kwargs
        super().save(self.directory.append("_obfuscater.pkl"))

    def update_symlinks(self, directory: Optional[str] = None):
        p = P(directory) if directory is not None else self.directory
        for path in p.search("*", r=False):
            if path.is_symlink(): continue
            if path.is_dir():
                try:
                    _ = self.real_to_phony[path.relative_to(self.directory).as_posix()]
                except KeyError:  # directory has not been obfuscated before
                    path_parent_r_obf = self.real_to_phony[path.parent.relative_to(self.directory).as_posix()]
                    self._add_map(path, path_r_obf=path_parent_r_obf)
                self.update_symlinks(directory=str(path))
                continue
            self._add_map(path, path_r_obf=self.real_to_phony[path.parent.relative_to(self.directory).as_posix()])
            phony = self.obfuscate(path)
            path.move(path=phony)
            path.symlink_to(phony)
        # raise NotImplementedError


def main():
    P("~/data/rst").copy(name="tst", overwrite=True)
    o = Obfuscator("~/data/tst")
    o.build_map()
    o.execute_map()
    o.save()
    # o.execute_map(forward=False)
    o.symlink_to_phoney()
    P("~/data/tst_obfuscater.Obfuscator.pkl").copy(path=r"C:\Users\alex\data\tst\meta\lol\dat.pkl")
    o.update_symlinks()


if __name__ == '__main__':
    pass
