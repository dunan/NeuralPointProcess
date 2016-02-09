import sys
from collections import Counter

if __name__ == '__main__':
    event_prefix = sys.argv[1]

    cnt = Counter()
    event_train = '%s-train.txt' % event_prefix
    event_test = '%s-test.txt' % event_prefix
    with open(event_train, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            for w in line:
                cnt[w] += 1

    top = cnt.most_common(1)[0][0]
    print 'most commom marker is', top

    err_cnt = 0
    total = 0
    with open(event_test, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            total += len(line) - 1
            for i in range(1, len(line)):
                if top != line[i]:
                    err_cnt += 1

    print 'error_rate:', float(err_cnt) / float(total)
