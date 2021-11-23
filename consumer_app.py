import threading
from confluent_kafka import Consumer, KafkaError, KafkaException
from infer_kafka import get_prediction_image

import cv2
import numpy as np
import time

consumer_config = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'kafka-image_process',
    'enable.auto.commit': False,
    'default.topic.config': {'auto.offset.reset': 'earliest'}
}


class ConsumerThread:
    def __init__(self, config, topic, batch_size):
        self.config = config
        self.topic = topic
        self.batch_size = batch_size

    def read_data(self):
        consumer = Consumer(self.config)
        consumer.subscribe(self.topic)
        self.run(consumer, 0, [], [])

    def run(self, consumer, msg_count, msg_array, metadata_array):
        try:
            while True:
                msg = consumer.poll(0.5)
                if msg == None:
                    continue
                elif msg.error() == None:
                    nparr = np.frombuffer(msg.value(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    msg_array.append(img)

                    timestamp_image = msg.timestamp()[1]
                    image_name = msg.headers()[0][1].decode("utf-8")
                    metadata_array.append((timestamp_image, image_name))

                    msg_count += 1
                    if msg_count % self.batch_size == 0:
                        img_array = np.asarray(msg_array)
                        predictions = get_prediction_image(img_array, self.batch_size)

                        for metadata, label in zip(metadata_array, predictions):
                            timestamp_image, image_name = metadata
                            doc = {
                                "image_name": image_name,
                                "prediction": label,
                                "time_stamp": timestamp_image
                            }

                            print(doc)

                        consumer.commit(asynchronous=False)
                        msg_count = 0
                        metadata_array = []
                        msg_array = []

                elif msg.error().code() == KafkaError._PARTITION_EOF:
                    print('End of partition reached {0}/{1}'
                          .format(msg.topic(), msg.partition()))
                else:
                    print('Error occured: {0}'.format(msg.error().str()))

        except KeyboardInterrupt:
            print("Detected Keyboard Interrupt. Quitting...")
            pass

        finally:
            consumer.close()

    def start(self, numThreads):
        for _ in range(numThreads):
            t = threading.Thread(target=self.read_data)
            t.daemon = True
            t.start()
            while True: time.sleep(10)


if __name__ == "__main__":
    topic = ["datainput"]
    consumer_thread = ConsumerThread(consumer_config, topic, 1)