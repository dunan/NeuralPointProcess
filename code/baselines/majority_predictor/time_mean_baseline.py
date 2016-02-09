import sys
from collections import Counter
import math

if __name__ == '__main__':
    time_prefix = sys.argv[1]
    scale = float(sys.argv[2])

    cnt = Counter()
    time_train = '%s-train.txt' % time_prefix
    time_test = '%s-test.txt' % time_prefix

    train_cnt = 0
    s = 0
    with open(time_train, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            for i in range(1, len(line)):
                s += scale * (float(line[i]) - float(line[i - 1]))
            train_cnt += len(line) - 1

    mean_predictor = s / train_cnt
    print 'mean predictor is', mean_predictor

    mae = 0.0
    rmse = 0.0
    test_cnt = 0
    with open(time_test, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            test_cnt += len(line) - 1
            for i in range(1, len(line)):
                y = scale * (float(line[i]) - float(line[i - 1]))
                mae += abs(y - mean_predictor)
                rmse += (y - mean_predictor) ** 2

    mae = mae / test_cnt
    rmse = math.sqrt(rmse / test_cnt)

    print 'mae:', mae, 'rmse:', rmse
