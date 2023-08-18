import sys

DEBUG=True

def debug_print(message, exit=False):
    
    if DEBUG:
        print(message)
        
        if exit:
            sys.exit(0)