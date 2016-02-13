import sys
import getopt
import os
import numpy as np

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
        opts, args = getopt.getopt(sys.argv[1:], "e:t:p:n:s:")
    except getopt.GetoptError:
        print '-e <event file> -t <time file> -p <test percent> -n <num_sequences>'
        sys.exit(2)

    event_file = ''
    time_file = ''
    percent = 0
    for opt, arg in opts:
        if opt == '-e':
            event_file = arg
        elif opt == '-t':
            time_file = arg
        elif opt == '-p':
            percent = float(arg)

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
    print folder, e_file_name, t_file_name

    assert len(time_lines) == len(event_lines)
    for i in range(len(time_lines)):
        length = len(time_lines[i])
        test_len = int(length * percent)
        train_len = length - test_len
        test_len += 1

        if train_len < 2 or test_len < 2:
            print 'too short in #', i
            train_event.append(['1'])
            test_event.append(['1'])
            train_time.append(['1'])
            test_time.append(['1'])
        else:
            train_event.append(event_lines[i][:train_len])
            train_time.append(time_lines[i][:train_len])

            test_event.append(event_lines[i][train_len - 1 : ])
            test_time.append(time_lines[i][train_len - 1 :])

    save_sequence('%s/%s-train.txt' % (folder, t_file_name), train_time)
    save_sequence('%s/%s-test.txt' % (folder, t_file_name), test_time)
    save_sequence('%s/%s-train.txt' % (folder, e_file_name), train_event)
    save_sequence('%s/%s-test.txt' % (folder, e_file_name), test_event)
