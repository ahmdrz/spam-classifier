from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class Classifier:
    def __init__(self, classifier):
        self._selected_classifier = classifier

        if classifier == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=6)
        elif classifier == 'svm':
            self.classifier = SVC(gamma='auto', C=2.0)
        elif classifier == 'random-forest':
            self.classifier = RandomForestClassifier(n_estimators=6)
        elif classifier == 'naive-bayes':
            self.classifier = GaussianNB()
        elif classifier == 'decision-tree':
            self.classifier = DecisionTreeClassifier()
        else:
            raise Exception('not supported classifier')

    def fit(self, samples, labels):
        self.classifier.fit(samples, labels)

    def predict(self, samples):
        return self.classifier.predict(samples)

    def accuracy(self, samples, labels):
        return accuracy_score(labels, self.predict(samples))

    def confusion_matrix(self, samples, labels):
        return confusion_matrix(labels, self.predict(samples))
