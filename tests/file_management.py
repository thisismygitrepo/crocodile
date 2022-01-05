import tempfile

import pytest
from crocodile.file_management import *


class Test_P:

    @staticmethod
    def test_copy():
        folder = P.tmp(folder="__test__/test_folder")
        folder.joinpath("test_content").touch()
        file = P.tmp(file="__test__/test_file")
        file.write_text("file content")
        dest = P.tmp(folder="__test__/test_destination")

        folder.copy(path=dest / "folder_copy_path_passed")
        folder.copy(folder=dest / "folder_copy_folder_passed")
        folder.copy(folder=dest / "folder_copy_folder_passed_2", name="name_passed_after_folder")

        file.copy(path=dest / "file_copy_path_passed")
        file.copy(folder=dest / "file_copy_folder_passed")
        file.copy(folder=dest / "file_copy_folder_passed_2", name="name_passed_after_folder")

    @staticmethod
    def test_zip():
        folder = P.tmp(folder="__test__/test_folder")
        folder.joinpath("test_content").touch()

        folder.zip()