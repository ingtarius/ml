
import sys

NUMBER = sys.argv[1]

if (int(NUMBER) % 2) == 0:
    print("%s;1" % NUMBER)
else:
    print("%s;0" % NUMBER)
