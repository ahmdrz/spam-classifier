import argparse
import arff
import os
import sys
import time
import numpy as np
from classifier.visualize import draw_chart
from random import shuffle
from classifier import Classifier
from classifier.cross_validation import kfold_cross_validation

classifiers = [
    "knn",
    "svm",
    "random-forest",
    "naive-bayes",
    "decision-tree",
]


def file_exist(file_path):
    return os.path.exists(file_path)


def read_dataset(file_path):
    with open(file_path) as handler:
        return arff.load(handler)


def parse_samples_labels(data):
    return data[:, :-1].astype(float), data[:, -1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="dataset in arff format",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--classifier",
        help="classifier algorithm",
        required=True,
    )
    parser.add_argument(
        "-k",
        "--kfold",
        help="k in kfold cross validation",
        default=10,
    )
    args = vars(parser.parse_args())

    if not file_exist(args["dataset"]):
        print("dataset file does not exist.")
        sys.exit(1)

    arg_classifiers = classifiers.copy()
    arg_classifiers.append("all")

    if args["classifier"] not in arg_classifiers:
        print("invalid type of classifier. this program only support '{}'.".format(
            ", ".join(classifiers)))
        sys.exit(2)

    return args


if __name__ == "__main__":
    args = parse_args()

    print("~ Reading dataset file ...")
    dataset = read_dataset(args["dataset"])
    data = dataset["data"]
    shuffle(data)
    data = np.array(data)
    print("- Dataset relation: '{}'.".format(dataset["relation"]))
    print("- Dataset size: {}.".format(len(data)))

    selected_classifiers = classifiers if args["classifier"] == 'all' else [
        args["classifier"]]

    all_accuracies_list = []
    all_labels = []
    for current_classifier in selected_classifiers:
        print("~ Running program with {} classifier.".format(current_classifier))
        classifier = Classifier(current_classifier)

        accuracy_list = []
        start_time = time.time()
        for i, (train, test) in enumerate(kfold_cross_validation(data, k=int(args["kfold"]))):
            samples, labels = parse_samples_labels(train)
            classifier.fit(samples, labels)

            samples, labels = parse_samples_labels(test)
            accuracy = classifier.accuracy(samples, labels)
            print("+ [{:02d}/10] accuracy: {:3d}%".format(i+1, int(accuracy*100)))

            accuracy_list.append(accuracy)

        all_accuracies_list.append(accuracy_list)
        all_labels.append(current_classifier)

        print("- Took {:.2f} seconds.".format(time.time() - start_time))

    print("~ Drawing chart to visualize accuracy list ...")

    draw_chart(
        range(len(all_accuracies_list[0])),
        all_accuracies_list,
        y_labels=all_labels,
        title='The accuracy graph for {} classifier(s).'.format(
            current_classifier)
    )
