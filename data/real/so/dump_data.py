from __future__ import print_function
import psycopg2 as pg
import getpass as G
from collections import namedtuple

output_events = 'events.txt'
output_time = 'time.txt'
output_userids = 'userids.txt'
output_badge_labels = 'badges.csv'

SO_events = namedtuple('SO_events', ['times', 'events', 'badge_map', 'userids'])

def write():
    try:
        with pg.connect(database='stackexchange', user='utkarsh', password=G.getpass('DB password: '), host='psql-science') as conn:
            cur = conn.cursor()
            # Ordering is important for mapping results back to the data, if needed.
            cur.execute('''SELECT userid, BadgeNames, Timestamp FROM so_data ORDER BY userid''')

            badge_map = {}
            badge_count = 1
            userids = []

            with open(output_events, 'w') as f_events, open(output_time, 'w') as f_time:
                for row in cur:
                    userid, events, times = row[0], row[1], row[2]
                    if len(set(times)) != len(times):
                        # If there are any repeated events, just skip the user.
                        continue

                    userids.append(userid)
                    event_ids = []
                    for badge in events:
                        if badge not in badge_map:
                            badge_map[badge] = badge_count
                            badge_count += 1
                        event_ids.append(badge_map[badge])

                    f_events.write(' '.join(str(x) for x in event_ids) + '\n')
                    # Can change times to something more granular than seconds.
                    f_time.write(' '.join(str(x) for x in times) + '\n')

            with open(output_userids, 'w') as f_userids:
                f_userids.write('userid\n')
                f_userids.writelines([str(x) + '\n' for x in userids])

            with open(output_badge_labels, 'w') as f_badges:
                f_badges.write('id, badge\n')
                for badge in badge_map:
                    f_badges.write('{}, {}\n'.format(badge_map[badge], badge))
    except pg.OperationalError:
        print('Not running on DB.')

def read_events():
    with open(output_events) as f_events:
        events = [[int(y) for y in x.split()] for x in f_events]

    with open(output_time) as f_times:
        times = [[float(y) for y in x.split()] for x in f_times]

    with open(output_userids) as f_userids:
        next(f_userids)
        userids = [int(x) for x in f_userids]

    badge_map = {}
    with open(output_badge_labels) as f_badge_labels:
        next(f_badge_labels)
        for row in f_badge_labels:
            id, name = row.split(',')
            badge_map[int(id)] = name.strip()

    return SO_events(events=events, times=times, badge_map=badge_map, userids=userids)

if __name__ == '__main__':
    write()


