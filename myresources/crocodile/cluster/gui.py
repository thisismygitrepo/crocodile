
# from dataclasses import dataclass
# import crocodile.toolbox as tb

# # TODO: ask chatgpt to write it such that Gooey takes in list of requiored and optionals directory without argparse hack
# def get_from_gui(dc_obj: dataclass):
#     res = f"""
# from gooey import Gooey

# @Gooey
# def main():
#     import argparse
#     argparser = argparse.ArgumentParser()

# """
#     for key, val in dict(dc_obj):
#         res += f"""
#     argparser.add_argument('--{key}', type={type(val).__name__}, default={val})
# """
#     res += f"""

#     args = argparser.parse_args()
#     return args
# """
#     file = tb.P.tmpfile(name="gui", suffix=".py")
