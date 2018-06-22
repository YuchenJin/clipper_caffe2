import os
import sys
from generator import *

def run_test(datapath, rate, duration, output, app_id):
    dataset = Dataset(datapath)
    gen = Generator(dataset, output, app_id)
    gen.run(rate, duration)
    gen.output_latencies(output)

def eval_face(sla, rps, n):
    def run(rps):
        print('Test rps %s' % rps)
        threads = []
        outputs = []
        for i in range(n):
            output = 'logs/{}models/vgg_face{}_sla{}_rate{}.txt'.format(n, i+1, sla, rps)
            t = Thread(target=run_test, args=(datapath, rps, duration, output, str(i+5)))
	    t.daemon = True
            threads.append(t)
            outputs.append(output)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
    duration = 100
    datapath = './resized_images/'
    print(datapath)
    print('Latency sla: %s ms' % sla)
    print("Number of models: %s" % n)
    run(rps)

def main():
    sla = float(sys.argv[1])
    rps = float(sys.argv[2])
    n = int(sys.argv[3])

    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    eval_face(sla, rps, n)

    #dataset = Dataset('./resized_images/')
    #output = 'logs/4models/vgg_face1_sla{}_rate{}.txt'.format(sla, rate)
    #test = ThroughputTest(dataset, output)
    #duration = 100
    #test.run(rate, duration)
    #test.output_latencies(output)

if __name__ == "__main__":
    main()
