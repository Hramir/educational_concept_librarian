import sys

class Logger(object):
    def __init__(self, log_file_name):
        self.terminal = sys.stdout
        self.log = open(log_file_name, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 