import os
import sys
from generator import *


def run_test(datapath, rate, duration, output, app_id, car_output, face_output):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id, car_output, face_output)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_inception(sla, n):
    def run(rps, iteration):
        print('Test rps %s' % rps)
        threads = []
        outputs = []
	car_outputs = []
	face_outputs = []
        for i in range(int(n)):
            output = 'logs/mobilenet{}_iter{}_sla{}_rate{}.txt'.format(i+1, iteration, sla, rps)
	    car_output = 'logs/car{}_iter{}_sla{}_rate{}.txt'.format(i+1, iteration, sla, rps)
            face_output = 'logs/face{}_iter{}_sla{}_rate{}.txt'.format(i+1, iteration, sla, rps)
	    if sla == 50:
	        app_id_base = 1
	    elif sla == 200:
	        app_id_base = 9
	    elif sla == 1000:
	        app_id_base = 13
	    else:
		print("Invalid sla")
		sys.exit()
            t = Thread(target=run_test, args=(datapath, rps, duration, output, str(i+app_id_base), car_output, face_output))
	    t.daemon = True
            threads.append(t)
            outputs.append(output)
	    car_outputs.append(car_output)
	    face_outputs.append(face_output)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
	aggr_good = 0
	aggr_total = 0
	kk = True
	for i, output in enumerate(outputs):
            good, total = parse_result(output)
	    percent = float(good) / total
            print('Mobilenet App %s: %.2f%%' % (i+1, percent*100))
            aggr_good += good
            aggr_total += total
        if float(aggr_good) / aggr_total < .99:
            kk = False

	for i, output in enumerate(car_outputs):
            good, total = parse_result(output)
	    percent = float(good) / total
            print('Car App %s: %.2f%%' % (i+1, percent*100))
            aggr_good += good
            aggr_total += total
        if float(aggr_good) / aggr_total < .99:
            kk = False

	for i, output in enumerate(face_outputs):
            good, total = parse_result(output)
	    percent = float(good) / total
            print('Person App %s: %.2f%%' % (i+1, percent*100))
            aggr_good += good
            aggr_total += total
        if float(aggr_good) / aggr_total < .99:
            kk = False

	if kk == True:
            return True
        
    duration = 10
    datapath = './resized_images/jackson_day/'
    print(datapath)
    print('Latency sla: %s ms' % sla)
    print("Number of models: %s" % n)
    #run(rps)

    rps = 10
    while True:
	for i in range(5):
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
    eval_inception(50, 2)

if __name__ == "__main__":
    main()
