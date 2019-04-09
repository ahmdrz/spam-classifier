### Classification algorithms (SpamClassifier)

> Here are our exercises of implementing classification algorithms in Python using sci-kit learn. 

It written in Python-3.6.7. Dependencies are available in `requirements.txt` file. You may have to install tkinter. Follow this instruction:

```
$ # on the debian-based OS like Ubuntu
$ sudo apt-get install python3-tk
```

<img width=100% src="https://github.com/ahmdrz/spam-classifier/raw/master/resources/spam-classifier.png">

Image is from [developers.google.com](https://developers.google.com/machine-learning/guides/text-classification/images/TextClassificationExample.png)

You can see `DOCUMENT.md` for more information.

#### Docker

To run this program without installing python3 and other libraries/dependencies, you can run our docker image.

```bash
$ docker pull ahmdrz/spamclassifier:latest
$ docker run ahmdrz/spamclassifier:latest
```

#### Dataset

> We used standard dataset named `spambase`. You can find it in dataset directory of our repository. This program support all of `arff` datasets that the class-label is in the last column.

#### Algorithms

1. kNN
2. Naive bayes
3. Decision tree
4. SVM
5. Random forest

TODO: **With neural-networks**

#### Results

The result contains the confusion matrix and the accuracy of each algorithm and will be available in the results directory.

Accuracy graph:

<img width=100% src="https://github.com/ahmdrz/spam-classifier/raw/master/resources/figure-1.png">

The configuration of each classifier listed below

1. n_neighbors in kNN: 6
2. C in SVC: 2.0
3. n_estimators in RandomForest: 6
4. all others were in the default configuration.

Confusion matrix for kNN with k=6:

<img width=100% src="https://github.com/ahmdrz/spam-classifier/raw/master/resources/confusion-matrix.png">

We used [confusion_matrix_pretty_print.py](https://github.com/wcipriano/pretty-print-confusion-matrix/blob/master/confusion_matrix_pretty_print.py) to generate this figure.

#### Authors

1. Nastaran Kiani (`no github address`)
2. Ahmadreza Zibaei ([@ahmdrz](https://github.com/ahmdrz))
