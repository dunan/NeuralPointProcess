import psycopg2 as pg
import getpass as G

output_events = 'events.txt'
output_time = 'time.txt'

with pg.connect(database='stackexchange', user='utkarsh', password=G.getpass('DB password: '), host='psql-science') as conn:
    cur = conn.cursor()
    # Ordering is important for mapping results back to the data, if needed.
    cur.execute('''SELECT EventIds, Timestamp FROM so_data ORDER BY userid''')

    with open(output_events, 'w') as f_events, open(output_time, 'w') as f_time:
        for row in cur:
            events, times = row[0], row[1]
            f_events.write(','.join(str(x) for x in events) + '\n')
            # Can change times to something more granular than seconds.
            f_time.write(','.join(str(x) for x in times) + '\n')
