import os
import sys
from generator import *

def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_game(sla, app_list):
    def run(max_rps):
        print('Test max rps %s' % max_rps)
        threads = []
        outputs = []
        for i in range(len(app_list)):
            output = 'logs/resnet{}_sla{}_rate{}.txt'.format(i+1, sla, max_rps)
	    rps = max_rps * float(app_list[i])/0.976516345296
	    print('App%d rps: %f' %(i+1, rps))
	    if sla == 50:
	        app_id_base = 1
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
    lenet_datapath = './resized_images_lenet/'
    print(datapath)
    print('Latency sla: %s ms' % sla)
    print("Number of models: %s" % len(app_list))
    #run(rps)

    rps = 10 
    #run(rps)
    while True:
	for i in range(3):
            good = run(rps)
            if good:
                break
	if good:
            rps += 10
	else:
	    break
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
    if len(sys.argv) > 1:
    	app_list = []
    	for element in sys.argv[1:]:
    	    app_list.append(element)
    else:
	print "usage %s element1 element2 [element3 ...]" % sys.argv[0]
    	sys.exit(1)

    print app_list

    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    eval_game(50, app_list)

if __name__ == "__main__":
    main()
