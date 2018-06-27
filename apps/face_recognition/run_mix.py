import os
import sys
from generator import *


def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_face(num_round):
    def run(rps):
        print('Test rps %s' % rps)
        threads = []
        outputs = []

        output = 'logs_mixed/round{}_vgg_face1_sla50_rate{}.txt'.format(num_round, rps)
        t = Thread(target=run_test, args=(datapath, rps, duration, output, '1'))
	t.daemon = True
        threads.append(t)
        outputs.append(output)

        output = 'logs_mixed/round{}_vgg_face2_sla100_rate{}.txt'.format(num_round, rps)
        t = Thread(target=run_test, args=(datapath, rps, duration, output, '6'))
	t.daemon = True
        threads.append(t)
        outputs.append(output)

        output = 'logs_mixed/round{}_vgg_face3_sla200_rate{}.txt'.format(num_round, rps)
        t = Thread(target=run_test, args=(datapath, rps, duration, output, '11'))
	t.daemon = True
        threads.append(t)
        outputs.append(output)

        #for i in range(n):
        #    output = 'logs_mixed/vgg_face{}_sla{}_rate{}.txt'.format(i+1, sla, rps)
	#    if sla == 50:
	#        app_id_base = 1
	#    elif sla == 100:
	#        app_id_base = 5
	#    elif sla == 200:
	#        app_id_base = 9
	#    else:
	#	print("Invalid sla")
	#	sys.exit()
        #    t = Thread(target=run_test, args=(datapath, rps, duration, output, str(i+app_id_base)))
	#    t.daemon = True
        #    threads.append(t)
        #    outputs.append(output)
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

    for i in range(1, 2):
    	eval_face(i)
    #eval_face(50, 1)
    #eval_face(100, 1)
    #eval_face(200, 1)
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
