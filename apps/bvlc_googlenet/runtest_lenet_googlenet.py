import os
import sys
from generator import *
import generator_lenet

def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def run_test_lenet(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = generator_lenet.Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_game(sla, app_list):
    def run(max_rps):
        print('Test max rps %s' % max_rps)
        threads = []
        outputs = []
        for i in range(len(app_list)):
	    if i == len(app_list) - 1:
            	output = 'logs/lenet1_sla{}_rate{}.txt'.format(sla, max_rps)
	    	rps = max_rps * 24.577161584478937 
	    	print('Lenet App%d rps: %f' %(1, rps))
            	t = Thread(target=run_test_lenet, args=(lenet_datapath, rps, duration, output, "1"))
	    	t.daemon = True
            	threads.append(t)
            	outputs.append(output)

	    else:
            	output = 'logs/googlenet{}_sla{}_rate{}.txt'.format(i+1, sla, max_rps)
	    	rps = max_rps * float(app_list[i])/4.83375591
	    	print('App%d rps: %f' %(i+1, rps))
	    	if sla == 50:
	    	    app_id_base = 1
	    	elif sla == 100:
	    	    app_id_base = 5
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
    lenet_datapath = './resized_images_lenet/'
    print(datapath)
    print('Latency sla: %s ms' % sla)
    print("Number of models: %s" % len(app_list))
    #run(rps)

    rps = 10 
    #run(rps)
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
    #eval_game(100, app_list)
    #eval_game(200, app_list)

if __name__ == "__main__":
    main()
