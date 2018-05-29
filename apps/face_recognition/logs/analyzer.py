import sys
import glob

print(glob.glob("./*txt"))


for i in glob.glob("./*txt"):
    with open(i) as f:
        print i
        success, total = 0, 0
        for line in f:
            total += 1
            if "False" in line:
                success += 1
        print(success*100.0/total)
