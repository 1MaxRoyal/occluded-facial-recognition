import sys
import datetime
from typing import Literal

now = datetime.datetime.now()
dt = now.strftime("%d%m%Y_%H%M")

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

def redirect_console_output(file_name):
    current_stdout = sys.stdout   
    sys.stdout = Logger(file_name)
    return current_stdout

# redirect stdout to text file
_MODES = Literal['train', 'test']
def log(mode):
    fileName = "logs/train_" if mode == 'train' else "logs/test_"
    current_stdout = redirect_console_output(fileName+dt+".txt")
    return current_stdout

def end(current_stdout):
    sys.stdout = current_stdout
    print("Done!")