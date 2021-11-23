from glob import glob
import concurrent.futures
from confluent_kafka import Producer
from skimage import io, color
from kafka_utils import *

producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'enable.idempotence': True,
    'acks': 'all',
    'retries': 100,
    'max.in.flight.requests.per.connection': 5,
    'compression.type': 'snappy',
    'linger.ms': 5,
    'batch.num.messages': 32
    }


class ProducerThread:
    def __init__(self, config):
        self.producer = Producer(config)

    def publishImage(self, image_path):
        img = color.rgb2gray(io.imread(image_path))
        img_name = str(image_path.split('/')[-1])
        frame_bytes = serializeImg(img)
        self.producer.produce(
            topic="datainput",
            value=frame_bytes,
            on_delivery=delivery_report,
            headers={
                "image_name": str.encode(img_name)
            }
        )
        self.producer.poll(0)
        return

    def start(self, img_paths):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.publishImage, img_paths)

        self.producer.flush()
        print("Finished...")


if __name__ == "__main__":
    img_dir = "/home/tanay/interviews/vector_ai/fashionMNIST/test_images/"
    img_paths = glob(img_dir + "*.jpeg")  # change extension here accordingly

    print(img_paths)
    producer_thread = ProducerThread(producer_config)
    producer_thread.start(img_paths)