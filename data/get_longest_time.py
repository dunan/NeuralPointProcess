import sys

if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    lines = f.readlines()
    lines = [l.strip().split(' ') for l in lines]

    max_len = len(lines[0])
    best = 0
    for i in range(1, len(lines)):
        if len(lines[i]) > max_len:
            max_len = len(lines[i])
            best = i
    f.close()

    with open('%s-longest' % sys.argv[1], 'w') as f:
        for item in lines[best]:
            f.write(item + '\n')

