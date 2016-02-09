import sys
import numpy as np

if __name__ == '__main__':
    time_train = sys.argv[1]
    scale = float(sys.argv[2])

    seq = []
    with open(time_train, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            for i in range(1, len(line)):
                seq.append(scale * (float(line[i]) - float(line[i - 1])))

    print 1.0 / np.var(seq)
