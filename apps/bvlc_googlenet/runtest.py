import os
import sys
from generator import *


def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_game(sla, app_list):
    def run(max_rps, iteration):
        print('Test max rps %s' % max_rps)
        threads = []
        outputs = []
        for i in range(len(app_list)):
            output = 'logs/googlenet{}_sla{}_rate{}_iter{}.txt'.format(i+1, sla, max_rps, iteration)
	    rps = max_rps
	    #rps = max_rps
	    print('App%d rps: %f' %(i+1, rps))
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
    print("Number of models: %s" % len(app_list))
    #run(rps)

    rps = 80 
    while True:
	for i in range(2):
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
            good = run(rps,2)
	    if not good:
                good = run(rps,3)
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
    if len(sys.argv) > 1:
    	app_list = []
    	for element in sys.argv[1:]:
    	    app_list.append(element)
    else:
	print "Usage %s app1 app2 [app3 ...]" % sys.argv[0]
    	sys.exit(1)

    print app_list

    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    eval_game(100, app_list)
    #eval_game(100, app_list)
    #eval_game(200, app_list)

if __name__ == "__main__":
    main()
