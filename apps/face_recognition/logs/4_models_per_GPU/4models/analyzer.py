import sys
import glob

logs = []
for i in range(1,5):
    files = glob.glob("./vgg_face{}_sla100*txt".format(str(i)))
    files.sort(key=lambda f: float(f.split("_")[-1].replace("rate", '').replace(".txt", '')))
    logs.append(files)

for i in range(0,len(logs[0])):
    result = []
    for k in range(0,4):
#        print logs[k][i]
        with open(logs[k][i]) as f:
            success, total = 0, 0
            for line in f:
                total += 1
                if "False" in line:
                    success += 1
            result.append(str(success*100.0/total))
    print " ".join(result)
