
# import pytest
# 


# class TestClass(tb.Base):
#     def __init__(self, arg1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.arg1 = arg1  # compulsary argument.

#     def add_one(self):
#         return self.arg1 + 1


# def test_base():
#     inst = TestClass(22)
#     save_path = tb.tmp("test_tmp").joinpath("saved_class.pkl")
#     inst.save_pickle(path=save_path, itself=True)  # only saved the data but not the code.
#     inst = TestClass.from_saved(path=save_path)  # load up without passing compulsay arguments.


# if __name__ == '__main__':
#     pass
