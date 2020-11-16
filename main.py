import math
import numpy as np
import pandas as pd
import csv
from collections import Counter


class SpamDetection:

    def __init__(self, PATH, LABEL):
        self.PATH = PATH
        self.LABEL = LABEL

        # 5572 SMS texts labeled as either ham or spam
        # Each row is an SMS
        # Column values denote tokens, separated by commas.

        self.tokens = self.read_data(self.PATH)

        with open(self.LABEL, 'r') as csvfile:
            self.line = csv.reader(csvfile)
            self.labels = list(self.line)[:-1]

        self.vocabulary = {}
        self.wordcount = np.zeros((len(self.tokens), len(self.vocabulary)))

    def read_data(self, path):
        """
        Method to read the given data.
        Separate each word and return an array of string sms' with separated words
        :param path: Path of the file taken from the user
        :return: Processed array of strings
        """
        data = pd.read_csv(path, dtype=None, sep='\t', header=None)
        tokens = data.values
        z = []
        for j in range(len(tokens)):
            for i in tokens[j]:
                # Replacing "," and then splitting
                z.append(i.replace(",", " ").split())
        return z

    def create_m_array(self):
        """
        This method creates the requested M array with the shape of N-by-d
        where N is the row number (number of SMS) and d is the feature no.
        It inspects and finds the frequency of each feature for each SMS.
        It constructs and saves the feature_set.csv file
        :return: Array of the frequency of each word.
        """

        # SMS will have each vocab's frequency
        SMS = self.tokens.copy()

        # Get the vocabulary. Each unique word
        vocabulary = set(word for msg in SMS for word in msg)

        self.vocabulary = vocabulary
        wordcounts = np.zeros((len(self.tokens), len(vocabulary)))

        # Count each unique word for each sms
        row_count = 0
        for sentence in self.tokens:
            bag_vector = np.zeros(len(vocabulary))
            for w in sentence:
                for i, word in enumerate(vocabulary):
                    if word == w:
                        bag_vector[i] += 1
            wordcounts[row_count] = bag_vector
            row_count += 1
        self.wordcount = wordcounts

        # Write data into a csv file
        data = pd.DataFrame(wordcounts)
        data.to_csv("feature_set.csv", index=False, header=False)
        return wordcounts

    def split_data(self, data):
        """
        Take the dataset and divide it into two (train and test).
        Train set will have the 80% of the given data set.
        :param data: Takes a dataset to divide from the user.
        :return: train: First 80% of the dataset to train.
        :return: test: Last 20% of the dataset to validate the model.
        :return: train_labels Labels of the train set.
        :return: test_labels: Labels of the test set.
        """
        split = 0.8
        train_size = int(len(data) * split)

        labels = self.labels

        train = data[0:train_size]
        train_labels = labels[0:train_size]
        test = data[train_size:]
        test_labels = labels[train_size:]

        return train, test, train_labels, test_labels

    def var_arrangement(self, train, train_labels):
        """Method to label spam and ham messages and found the values that will be needed"""
        # Take the count of SMSs', spams and hams. Also, declare alpha array for laplace smoothing
        N = len(train)
        N_spam = 0
        N_ham = 0

        # Find N, N_ham and N_spam
        for i in range(len(train_labels)):
            if train_labels[i] == ['1']:
                N_spam = N_spam + 1
            else:
                N_ham = N_ham + 1

        # Pi spam and Pi ham
        pi_spam = N_spam / N
        pi_ham = N_ham / N

        # Array declaration to store spam and ham sms' in two different arrays
        spams = np.zeros((N_spam, len(test[0])))
        hams = np.zeros((N_ham, len(test[0])))

        # s and h for indexing of spam and ham arrays
        s = 0
        h = 0
        # Loop to separate spam and ham sms'
        for i in range(len(train_labels)):
            if train_labels[i] == ['1']:
                spams[s] = train[i]
                s += 1
            else:
                hams[s] = train[i]
                h += 1

        return spams, hams, pi_spam, pi_ham

    def model(self, test, spams, hams):
        """
        Method to train and validate the set with a Multinomial Naive Bayes Model.
        Both the question 2.2 and 2.3 is applied in this method simultaneously.
        Laplace Smoothing variables and arrays indicated with "lap" prefix.
        :param train: Dataset to train the model.
        :param train_labels: Labels of the train set.
        :param test: Dataset to validate the model.
        :param test_labels: Labels of the test set.
        """

        alpha = np.ones(len(test[0]))

        # Find tjs' that is number of occurrences
        tj_spam = np.sum(spams, axis=0)
        tj_ham = np.sum(hams, axis=0)

        # Laplace smoothing Tjs'
        lap_tj_spam_x = np.add(spams, alpha)
        lap_tj_spam = np.sum(lap_tj_spam_x, axis=0)
        lap_tj_ham_x = np.add(hams, alpha)
        lap_tj_ham = np.sum(lap_tj_ham_x, axis=0)

        # Sum of all no. of words in spam/ham
        sum_tj_spam = tj_spam.sum()
        sum_tj_ham = tj_ham.sum()

        # Laplace sum of tjs'
        lap_sum_tj_spam = sum_tj_spam + (len(self.vocabulary) * alpha[0])
        lap_sum_tj_ham = sum_tj_ham + (len(self.vocabulary) * alpha[0])

        # arrays for probabilities
        theta_spam = []
        lap_theta_spam = []
        theta_ham = []
        lap_theta_ham = []
        for i in range(len(tj_spam)):
            prob = float((tj_spam[i])/sum_tj_spam)
            lap_prob = float((lap_tj_spam[i])/lap_sum_tj_spam)
            if prob != 0:
                prob = float(math.log(prob))
            else:
                prob = 0
            theta_spam.append(prob)
            lap_theta_spam.append(math.log(lap_prob))

            prob_ham = float((tj_ham[i])/sum_tj_ham)
            lap_prob_ham = float((lap_tj_ham[i]) / lap_sum_tj_ham)
            if prob_ham != 0:
                prob_ham = float(math.log(prob_ham))
            else:
                prob_ham = 0
            theta_ham.append(prob_ham)
            lap_theta_ham.append(math.log(lap_prob_ham))

        return theta_spam, theta_ham, lap_theta_ham, lap_theta_spam, pi_ham, pi_spam

    def train(self, test, test_labels, theta_spam, theta_ham, lap_theta_ham, lap_theta_spam, pi_ham, pi_spam):

        pred = 0
        lap_pred = 0
        for row in range(len(test)):

            # First training
            total_theta_spam = np.multiply(theta_spam, test[row])
            total_theta_ham = np.multiply(theta_ham, test[row])

            # Laplace Smoothing Part
            lap_total_theta_spam = np.multiply(lap_theta_spam, test[row])
            lap_total_theta_ham = np.multiply(lap_theta_ham, test[row])

            # First training
            total_theta_spam = total_theta_spam.sum()
            total_theta_ham = total_theta_ham.sum()

            # Laplace Smoothing Training Part
            lap_total_theta_spam = lap_total_theta_spam.sum()
            lap_total_theta_ham = lap_total_theta_ham.sum()

            # Prediction
            pred_spam = math.log(pi_spam) + total_theta_spam
            pred_ham = math.log(pi_ham) + total_theta_ham

            # Laplace Smoothing prediction
            lap_pred_spam = math.log(pi_spam) + lap_total_theta_spam
            lap_pred_ham = math.log(pi_ham) + lap_total_theta_ham

            # Validate predictions
            if pred_spam >= pred_ham:
                prediction = ['1']
            else:
                prediction = ['0']

            if prediction == test_labels[row]:
                pred = pred + 1
            else:
                pred

            # Laplace Check Prediction Part
            if lap_pred_spam >= lap_pred_ham:
                lap_prediction = ['1']
            else:
                lap_prediction = ['0']

            if lap_prediction == test_labels[row]:
                lap_pred = lap_pred + 1
            else:
                lap_pred

        accuracy = float(pred/len(test)) * 100
        print("Accuracy " + str(accuracy))
        print("Total set ", len(test))
        print("Correct predictions", pred)
        print("False predictions ", (len(test) - pred))
        np.savetxt("test_accuracy.csv", np.array([accuracy]))

        # Laplace Accuracy print
        lap_accuracy = float(lap_pred/len(test)) * 100
        print("------------------")
        print("Laplace Accuracy " + str(lap_accuracy))
        print("Correct Laplace predictions", lap_pred)
        print("False Laplace predictions ", (len(test) - lap_pred))
        np.savetxt("test_accuracy_laplace.csv", np.array([lap_accuracy]))

    # def feature_vocab(self):
    #     # Feature Selection Vocab.
    #     c = Counter([x for sublist in self.tokens for x in sublist])
    #     Vr_count = Counter(el for el in c.elements() if c[el] >= 10)
    #     feat_vocab = list(Vr_count)
    #     return feat_vocab
    #
    # def feature_selected_array_arrangement(self, feat_vocab):
    #     wordcounts = np.zeros((len(self.tokens), len(self.vocabulary)))
    #
    #     # Count each unique word for each sms
    #     row_count = 0
    #     for sentence in self.tokens:
    #         bag_vector = np.zeros(len(self.vocabulary))
    #         for w in sentence:
    #             for i, word in enumerate(feat_vocab):
    #                 if word == w:
    #                     bag_vector[i] += 1
    #         wordcounts[row_count] = bag_vector
    #         row_count += 1
    #     feat_words = wordcounts
    #     return feat_words
    #
    # def feature_selection(self, train, train_labels, test, test_labels):
    #     """
    #     Method to create the new filtered vocabulary with the counts of greater than 10
    #     :return: Vr, new vocabulary list. new_feature, count array of the new vocab list.
    #     """
    #     feats = []
    #     feature_set = self.feature_selected_array_arrangement(feats)
    #
    #
    #     # arrays for probabilities
    #     theta_spam = []
    #     lap_theta_spam = []
    #     theta_ham = []
    #     lap_theta_ham = []



if __name__ == '__main__':
    x = SpamDetection("tokenized_corpus.csv", "labels.csv")
    SMS_array = x.create_m_array()
    train, test, train_labels, test_labels = x.split_data(SMS_array)
    spams, hams, pi_spam, pi_ham = x.var_arrangement(train, train_labels)
    theta_spam, theta_ham, lap_theta_ham, lap_theta_spam, pi_ham, pi_spam = x.model(test, spams, hams)
    x.train(test, test_labels, theta_spam, theta_ham, lap_theta_ham, lap_theta_spam, pi_ham, pi_spam)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
