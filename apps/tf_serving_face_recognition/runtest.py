import os
import sys
from generator import *


def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_face(n):
    def run(rps):
        print('Test rps %s' % rps)
        threads = []
        outputs = []
        for i in range(n):
            output = 'logs/{}models/vgg_face{}_rate{}.txt'.format(n, i+1, rps)
            t = Thread(target=run_test, args=(datapath, rps, duration, output, i+1))
	    t.daemon = True
            threads.append(t)
            outputs.append(output)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
	aggr_good = 0
	aggr_total = 0
	for i, output in enumerate(outputs):
            good, total = parse_result(output)
	    percent = float(good) / total
            print('App %s: %.2f%%' % (i+1, percent*100))
            aggr_good += good
            aggr_total += total
        if float(aggr_good) / aggr_total < .99:
            return False
        return True
        
    duration = 100
    datapath = './resized_images/'
    print(datapath)
    print("Number of models: %s" % n)
    #run(rps)

    rps = 10
    while True:
        good = run(rps)
        if not good:
            break
        rps += 10
    for rps in np.arange(rps-9.5, rps, 0.5):
        good = run(rps)
        if not good:
            break

def parse_result(fn):
    total, good = 0, 0
    with open(fn) as f:
        for line in f:
	    total += 1
	    if "False" in line:
	        good += 1
    return good, total

def main():
    #sla = int(sys.argv[1])
    #rps = float(sys.argv[2])
    #n = int(sys.argv[3])

    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    eval_face(1)

if __name__ == "__main__":
    main()
