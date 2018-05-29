import os
import glob
import time
import random
import logging
from threading import Thread
from multiprocessing import Queue
from datetime import datetime
import requests, json, numpy as np
import base64


class Dataset(object):
    def __init__(self, data_dir, max_count=10000):
        self.images = []
        for fn in os.listdir(data_dir):
            with open(os.path.join(data_dir, fn), 'rb') as f:
                self.images.append(base64.b64encode(f.read()))
            if max_count > 0 and len(self.images) >= max_count:
                break

    def rand_idx(self):
        return random.randint(0, len(self.images) - 1)


class Worker(Thread):
    def __init__(self, idx, dataset, queue, output):
        super(Worker, self).__init__()
        self.daemon = True
        self.index = idx
        self.dataset = dataset
        self.queue = queue
        self.lats = []
        self.img_idx = random.randint(0, len(self.dataset.images) - 1)
        self.output= output


    def run(self):
        while True:
            idx = self.queue.get()
            if idx == -1:
                break
            img = self.dataset.images[self.img_idx]
            headers = {"Content-type": "application/json"}
            start = datetime.now()
            r = requests.post("http://localhost:1337/face-recognition1/predict", headers=headers, data=json.dumps({"input": img})).json()
            end = datetime.now()
            #print r
            with open(self.output, 'a') as fout:
                fout.write(str(r) + "\n")
            lat = (end - start).total_seconds() * 1000.0

            self.lats.append((self.img_idx, lat))
            self.img_idx = (self.img_idx + 1) % len(self.dataset.images)


class ThroughputTest(object):
    def __init__(self, dataset, output):
        self.dataset = dataset
        self.queue = Queue()
        self.workers = []
        for i in range(0, 500):
            worker = Worker(i, dataset, self.queue, output)
            worker.start()
            self.workers.append(worker)

    def run(self, rate, duration):
        count = 0
        gap = 1. / rate
        total = duration * rate
        beg = time.time()
        while True:
            now = time.time()
            while count * gap <= now - beg:
                self.queue.put(1)
                count += 1
                now = time.time()
                if count >= total:
                    break
            if count >= total or now - beg >= duration:
                break
            to_sleep = beg + count * gap - now
            if to_sleep > 0:
                time.sleep(to_sleep)
        elapse = time.time() - beg
        logging.info('Generate {} requests in {} sec, rate: {} req/s'.format(
            count, elapse, float(count) / elapse))

    def stop_all(self):
        for _ in range(len(self.workers)):
            self.queue.put(-1)
        for t in self.workers:
            t.join()

    def output_latencies(self, output):
        self.stop_all()
        #with open(output, 'a') as fout:
        #    for worker in self.workers:
        #        for img_idx, lat in worker.lats:
        #            fout.write('%s %s\n' % (img_idx, lat))
        logging.info('Output latencies to %s' % output)
