import os
import sys
from clipper_client2 import *


def main():
    sla = float(sys.argv[1])
    rate = float(sys.argv[2])

    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    dataset = Dataset('./resized_images/')
    output = 'logs/vgg_face2_sla{}_rate{}.txt'.format(sla, rate)
    test = ThroughputTest(dataset, output)
    duration = 100
    test.run(rate, duration)
    test.output_latencies(output)


if __name__ == "__main__":
    main()