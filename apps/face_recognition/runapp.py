import os
import sys
from clipper_client import *


def main():
    sla = int(sys.argv[1])
    rate = int(sys.argv[2])

    FORMAT = "[%(asctime)-15s %(levelname)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    dataset = Dataset('/hdfs/pnrsy/v-haicsh/datasets/vgg_face/')
    test = ThroughputTest(dataset)
    duration = 120
    test.run(rate, duration)
    test.output_latencies('vgg_face_sla{}_rate{}.txt'.format(sla, rate))


if __name__ == "__main__":
    main()
