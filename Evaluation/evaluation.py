#!/usr/bin/env python
from __future__ import division

import csv
import math
import os
import sys

from sklearn import metrics


def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix


def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1


def extract_field(truth, submission, index):
  gold = []
  guess = []
  for key, value in truth.items():
    gold.append(value[index])
    guess.append(submission[key][index])
  return gold, guess


def compute_scoreA(truth, submission):
  gold, guess = extract_field(truth, submission, 0)
  score = compute_f1(guess, gold)
  return score


def compute_scoreB(truth, submission):
  results = []
  total_occurences = 0
  for index in range(1, 5):
    gold, guess = extract_field(truth, submission, index)
    f1_score = compute_f1(guess, gold)
    weight = gold.count(True)
    total_occurences += weight
    results.append(f1_score * weight)
  return sum(results) / total_occurences


def main(argv):
  # as per the metadata file, input and output directories are the arguments
  [_, input_dir, output_dir] = argv
  # unzipped submission data is always in the 'res' subdirectory
  # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
  submission_file_name = 'answer.txt'
  submission_dir = os.path.join(input_dir, 'res')
  submission_path = os.path.join(submission_dir, submission_file_name)
  if not os.path.exists(submission_path):
    raise ValueError('Expected submission file %s, found files %s' %
                     (submission_file_name, os.listdir(submission_dir)))
  submission = {}
  rowsize = None
  with open(submission_path) as submission_file:
    reader = csv.reader(submission_file, delimiter='\t')
    count = 1
    for row in reader:
      if len(row) != 2 and len(row) != 6:
        raise ValueError(
            'Wrong number of columns in line %d, expected 2 or 6.' % count)
      if rowsize and len(row) != rowsize:
        raise ValueError('Inconsistent number of columns in line %d.' % count)
      rowsize = len(row)
      count += 1
      submission[row[0]] = [bool(int(x)) for x in row[1:]]

  # unzipped reference data is always in the 'ref' subdirectory
  # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
  truth = {}
  with open(os.path.join(input_dir, 'ref', 'truth.txt')) as truth_file:
    reader = csv.reader(truth_file, delimiter='\t')
    for row in reader:
      if len(row) != 6:
        raise ValueError('Wrong number of columns in reference file.')
      truth[row[0]] = [bool(int(x)) for x in row[1:]]

  # Check for any missing entries, to simplify scoring code.
  for key in truth.keys():
    if key not in submission:
      raise ValueError('missing element %s in submission' % key)

  # the scores for the leaderboard must be in a file named "scores.txt"
  # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
  with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    scorea = compute_scoreA(truth, submission)
    output_file.write('taska: {0}\n'.format(scorea))
    scoreb = 0.0
    if rowsize == 6:
      scoreb = compute_scoreB(truth, submission)
    output_file.write('taskb: {0}\n'.format(scoreb))



if __name__ == '__main__':
  try:
    main(sys.argv)
  except Exception as e:
    sys.exit(e)
