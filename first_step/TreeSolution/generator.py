
import sys

NUMBER = sys.argv[1]

#print("NUMBER;FEATURE;EVEN")

if (int(NUMBER) % 2) == 0:
    print("%s;%s;1" % (NUMBER, NUMBER[-1:]))
    #print("%s,%s;1" % (NUMBER, NUMBER))
else:
    print("%s;%s;0" % (NUMBER, NUMBER[-1:]))
    #print("%s,%s;1" % (NUMBER, NUMBER))
