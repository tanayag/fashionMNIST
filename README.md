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


