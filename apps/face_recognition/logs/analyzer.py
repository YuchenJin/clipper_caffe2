import sys
import glob

logs = glob.glob("./vgg_face1_sla100*txt")
logs.sort(key=lambda f: float(f.split("_")[-1].replace("rate", '').replace(".txt", '')))

for i in logs:
    with open(i) as f:
#        print i
        success, total = 0, 0
        for line in f:
            total += 1
            if "False" in line:
                success += 1
        print(success*100.0/total)
