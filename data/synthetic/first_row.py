import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    time_file = '%s/time.txt' % dataset
    with open(time_file, 'r') as f:
        line = f.readline().strip().split(' ')
        with open('%s/row1-time-test.txt' % dataset, 'w') as fout:
            for i in range(10000):
                fout.write(line[i] + ' ')
            fout.write('\n')
        with open('%s/row1-event-test.txt' % dataset, 'w') as fout:
            for i in range(10000):
                fout.write('0 ')
            fout.write('\n')
