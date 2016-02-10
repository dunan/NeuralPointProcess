import sys
from collections import Counter, defaultdict

def train_model(filename, order):
    assert order >= 1
    model = defaultdict(Counter)
    label_cnt = Counter()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            if len(line) <= order or len(line) <= 2:
                continue
            for w in line:
                label_cnt[w] += 1
            state = tuple(line[:order])
            for i in range(order, len(line)):
                model[state][line[i]] += 1
                state = state[1:] + (line[i],)
                
    for key in model:
        t = model[key].most_common(1)[0][0]
        model[key] = t

    return model, label_cnt

if __name__ == '__main__':
    event_prefix = sys.argv[1]
    order = int(sys.argv[2])
    event_train = '%s-train.txt' % event_prefix
    event_test = '%s-test.txt' % event_prefix

    model, label_cnt = train_model(event_train, order)
    top_pred = label_cnt.most_common(1)[0][0]
    total = 0
    err_cnt = 0
    with open(event_test, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
           
            total += len(line) - 1
            for i in range(1, order):
                if i >= len(line):
                    break
                if top_pred != line[i]:
                    err_cnt += 1

            state = tuple(line[:order])
            for i in range(order, len(line)):
                if state in model:
                    pred = model[state]
                else:
                    pred = top_pred
                state = state[1:] + (line[i],)

                if pred != line[i]:
                    err_cnt += 1
    
    print 'error rate:', float(err_cnt) / float(total), 'of', total


