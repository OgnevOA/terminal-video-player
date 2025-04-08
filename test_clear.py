import time
import sys

print("Screen 1")
time.sleep(2)
sys.stdout.write("\033[2J\033[H") # Clear + Home
sys.stdout.flush()
print("Screen 2")
time.sleep(2)