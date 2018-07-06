import os
import sys
from generator import *


def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_face(sla, n):
    def run(rps, iteration):
        print('Test rps %s' % rps)
        threads = []
        outputs = []
        for i in range(n):
            output = 'logs/lenet1_iter{}_{}_sla{}_rate{}.txt'.format(iteration, i+1, sla, rps)
   	    rps = rps * 4.0961935960077902 * 6
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
        
    duration = 50
    datapath = './resized_images/'
    print(datapath)
    print('Latency sla: %s ms' % sla)
    print("Number of models: %s" % n)
    #run(rps)

    rps = 20
    while True:
	for i in range(10):
	    good = run(rps, i)
	    if good:
	        break
        rps += 10
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
    #sla = int(sys.argv[1])
    #rps = float(sys.argv[2])
    #n = int(sys.argv[3])

    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    #eval_face(200, 2)
    #eval_face(50, 1)
    eval_face(100, 1)
    #eval_face(1000, 1)
    #eval_face(1000, 1)
    #eval_face(50, 2)
    #eval_face(100, 2)
    #eval_face(200, 2)
    #eval_face(50, 3)
    #eval_face(100, 3)
    #eval_face(200, 3)
    #eval_face(50, 4)
    #eval_face(100, 4)
    #eval_face(200, 4)

if __name__ == "__main__":
    main()
