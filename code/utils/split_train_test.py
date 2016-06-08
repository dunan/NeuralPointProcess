import sys
import getopt
import os
import numpy as np
import random

def load_file(filename):
    fid = open(filename, 'r')
    lines = fid.readlines()
    fid.close()
    data = []
    for line in lines:
        line = line.strip().split(' ')
        data.append(line)
    return data

def save_sequence(filename, sequence):
    with open(filename, 'w') as f:
        for seq in sequence:
            for s in seq:
                f.write(s + ' ')
            f.write('\n')

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "e:t:p:n:s:r:")
    except getopt.GetoptError:
        print '-e <event file> -t <time file> -p <test percent> -n <num_sequences>'
        sys.exit(2)

    event_file = ''
    time_file = ''
    percent = 0
    num_seqs = 0
    for opt, arg in opts:
        if opt == '-e':
            event_file = arg
        elif opt == '-t':
            time_file = arg
        elif opt == '-p':
            percent = float(arg)
        elif opt == '-n':
            num_seqs = int(arg)
        elif opt == '-s':
            shift = int(arg)
        elif opt == '-r':
            rr = int(arg)

    folder, e_file = os.path.split(event_file)
    e_file_name = os.path.splitext(e_file)[0]
    folder, t_file = os.path.split(time_file)
    t_file_name = os.path.splitext(t_file)[0]
    time_lines = load_file(time_file)
    event_lines = load_file(event_file)

    train_time = []
    train_event = []
    test_time = []
    test_event = []
    if len(time_lines) == 1: # only contains one sequence
        assert num_seqs > 0 # the sequence num should be given in this case
        seg_len = len(time_lines[0]) / num_seqs

        prev_time = 0
        for i in range(num_seqs):
            swapped = 0
            if random.random() > 0.5:
                test_len = int(seg_len * percent)
                train_len = seg_len - test_len
            else:
                swapped = 1
                train_len = int(seg_len * percent)
                test_len = seg_len - train_len

            train_event.append(event_lines[0][i * seg_len : i * seg_len + train_len])
            test_event.append(event_lines[0][i * seg_len + train_len : (i + 1) * seg_len])

            train_time.append(time_lines[0][i * seg_len : i * seg_len + train_len])
            test_time.append(time_lines[0][i * seg_len + train_len : (i + 1) * seg_len])

            if shift:
                for j in range(len(train_time[-1])):
                    train_time[-1][j] = str(float(train_time[-1][j]) - prev_time)

                prev_time = float(time_lines[0][i * seg_len + train_len - 1])
                for j in range(len(test_time[-1])):
                    test_time[-1][j] = str(float(test_time[-1][j]) - prev_time)
                if i < num_seqs - 1:
                    prev_time = float(time_lines[0][(i + 1) * seg_len - 1])

            if swapped:
                t = train_event[-1]
                train_event[-1] = test_event[-1]
                test_event[-1] = t
                t = train_time[-1]
                train_time[-1] = test_time[-1]
                test_time[-1] = t
    else:
        p = np.random.permutation(len(time_lines))
        num_test_seqs = int(len(p) * percent)
        num_train_seqs = len(p) - num_test_seqs
        
        for i in range(num_train_seqs):
            train_time.append(time_lines[p[i]])
            train_event.append(event_lines[p[i]])

        for i in range(num_test_seqs):
            test_time.append(time_lines[p[i + num_train_seqs]])
            test_event.append(event_lines[p[i + num_train_seqs]])

    print folder, e_file_name, t_file_name
    save_sequence('%s/%s-%d-train.txt' % (folder, t_file_name, rr), train_time)
    save_sequence('%s/%s-%d-test.txt' % (folder, t_file_name, rr), test_time)
    save_sequence('%s/%s-%d-train.txt' % (folder, e_file_name, rr), train_event)
    save_sequence('%s/%s-%d-test.txt' % (folder, e_file_name, rr), test_event)
