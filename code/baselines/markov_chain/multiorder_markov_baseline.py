import sys
from collections import Counter, defaultdict

def train_model(filename, order):
    assert order >= 1
    model = defaultdict(Counter)
    label_cnt = Counter()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            for w in line:
                label_cnt[w] += 1

            state = (line[0],)
            for i in range(1, len(line)):
                for j in range(len(state)):
                    model[state[j:]][line[i]] += 1
                if len(state) == order:
                    state = state[1:] + (line[i],)
                else:
                    state = state + (line[i],)
                
    for key in model:
        t = model[key].most_common(1)[0][0]
        model[key] = t

    return model, label_cnt

if __name__ == '__main__':
    event_prefix = sys.argv[1]
    order = int(sys.argv[2])
    assert order > 0
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

            state = (line[0],)
            for i in range(1, len(line)):
                assert len(state) <= order
                if state in model:
                    pred = model[state]
                else:
                    found = False
                    for j in range(1, len(state)):
                        if state[j:] in model:
                            found = True
                            pred = model[state[j:]]
                            break
                    if not found:
                        pred = top_pred

                if pred != line[i]:
                    err_cnt += 1

                if len(state) == order:
                    state = state[1:] + (line[i],)
                else:
                    state = state + (line[i],)
    
    print 'error rate:', float(err_cnt) / float(total)


