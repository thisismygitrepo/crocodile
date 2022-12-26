
import time


def expensive_function():
    print("Hello, I am an expensive function, and I just started running ...")
    a = 1 + 1
    time.sleep(2)
    print("I'm done, I crunched a lot of numbers.")
    return a


if __name__ == '__main__':
    pass
