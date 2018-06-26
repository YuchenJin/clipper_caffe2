import sys
import glob

logs = []
for i in range(1,int(sys.argv[1])+1):
    files = glob.glob("./vgg_face{}_sla{}*txt".format(str(i), sys.argv[2]))
    files.sort(key=lambda f: float(f.split("_")[-1].replace("rate", '').replace(".txt", '')))
    logs.append(files)

for i in range(0,len(logs[0])):
    result = []
    for k in range(0,int(sys.argv[1])):
        with open(logs[k][i]) as f:
	    rate = logs[k][i].split("_")[-1].replace("rate", '').replace(".txt", '')
	    if len(result) == 0:
	        result.append(rate)
            success, total = 0, 0
            for line in f:
                total += 1
                if "False" in line:
                    success += 1
            result.append(str(success*100.0/total))
    print " ".join(result)
