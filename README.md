# Classification Module using CNNs



### To Train: 

`python main.py -train t -data_dir_train ./image_data/train -data_dir_test ./image_data/test`


### To Infer:

`python main.py -predict t -data_dir_infer ./test_images/`

While Inference, go to file `class_maps.py`
and add a dictionary mapping for label to index.
For E.g.

```
CLASS_MAP = {
    "FashionMNIST": {
        0: "Ankle_Boot",
        1: "Bag",
        2: "Coat",
        3: "Dress",
        4: "Pullover",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "T_Shirt",
        9: "Trouser",
    },
    # Example;
    "DataSetName": {
        0: "label_name",
        1: "label_name"
        # so on...
    }
}
```
Remember to add in lexicographical order, since pytorch reads it that way.
Class map for FashionMNIST already added, add for the new dataset you train.

These are the basic parameters to train and test a model. More parameters present, check
`python main.py -h`

### Message Broker

1. Download and extract Kafka

`wget https://dlcdn.apache.org/kafka/3.0.0/kafka_2.13-3.0.0.tgz`

`tar -xzf kafka_2.13-3.0.0.tgz`

2. Go into the downloaded kafka folder

`cd kafka_2.13-3.0.0.tgz`

3. Start the zookeeper

`bin/zookeeper-server-start.sh config/zookeeper.properties`

4. Start Kafka

`bin/kafka-server-start.sh config/server.properties`

5. Run the script `producer_app.py`

`python producer_app.py`

6. Run the script `consumer_app.py`

`python consumer_app.py`

