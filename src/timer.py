from time import time

class Timer(object):
    def __enter__(self):
        self.__start = time()

    def __exit__(self, type, value, traceback):
        self.__finish = time()
        
        print(self.__finish - self.__start)

