import sys
import getopt
import os


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
        opts, args = getopt.getopt(sys.argv[1:], "e:t:p:n:")
    except getopt.GetoptError:
        print '-e <event file> -t <time file> -p <test percent>'
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
        test_len = int(seg_len * percent)
        train_len = seg_len - test_len

        for i in range(num_seqs):
            train_time.append(time_lines[0][i * seg_len : i * seg_len + train_len])
            train_event.append(event_lines[0][i * seg_len : i * seg_len + train_len])
            test_time.append(time_lines[0][i * seg_len + train_len - 1 : (i + 1) * seg_len])
            test_event.append(event_lines[0][i * seg_len + train_len - 1 : (i + 1) * seg_len])
    else:
        print 'hello'

    print folder, e_file_name, t_file_name
    save_sequence('%s/%s-train.txt' % (folder, t_file_name), train_time)
    save_sequence('%s/%s-test.txt' % (folder, t_file_name), test_time)
    save_sequence('%s/%s-train.txt' % (folder, e_file_name), train_event)
    save_sequence('%s/%s-test.txt' % (folder, e_file_name), test_event)
