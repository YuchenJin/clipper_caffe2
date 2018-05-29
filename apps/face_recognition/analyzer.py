import sys

total, success = 0, 0
slo = float(sys.argv[2])
with open(sys.argv[1]) as f:
    for line in f:
        lat = float(line.strip().split()[1])
        total += 1
        if lat <= slo:
            success += 1

print(success*100.0/total)
