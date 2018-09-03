import os
import sys
from generator import *


def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_inception(sla, n):
    def run(rps, iteration):
        print('Test rps %s' % rps)
        threads = []
        outputs = []
        for i in range(int(n)):
            output = 'logs/2models/inception{}_iter{}_sla{}_rate{}.txt'.format(i+1, iteration, sla, rps)
	    if sla == 100:
	        app_id_base = 1
	    elif sla == 200:
	        app_id_base = 9
	    elif sla == 1000:
	        app_id_base = 13
	    else:
		print("Invalid sla")
		sys.exit()
            t = Thread(target=run_test, args=(datapath, rps, duration, output, str(i+app_id_base)))
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
    print('Latency sla: %s ms' % sla)
    print("Number of models: %s" % n)
    #run(rps)

    rps = 40
    while True:
	for i in range(3):
	    good = run(rps, i)
	    if good:
	        break
	if good:
            rps += 10
	else:
	    break
    for rps in np.arange(rps-9.5, rps, 0.5):
        good = run(rps, 1)
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
    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    eval_inception(int(sys.argv[1]), int(sys.argv[2]))

if __name__ == "__main__":
    main()
