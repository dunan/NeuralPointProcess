import argparse
import pandas as pd
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('inputfile', help='File containing sentiment data.')

def readFile(inputfile):
    data = []
    with open(inputfile, 'r') as f:
        for row in f:
            userid, timestamp, sentiment = row.split()
            data.append({
                'userid'    : int(userid),
                'timestamp' : int(timestamp),
                'sentiment' : float(sentiment)
            })
    return pd.DataFrame.from_dict(data)

datasets = {
        'fight': readFile('Sentiment_FIghtTwitter.txt'),
        'movie': readFile('Sentiment_MovieTwitter.txt'),
        'bolly': readFile('Sentiment_BollywoodTwitter.txt'),
        'politics': readFile('Sentiment_PoliticsTwitter.txt')
}

for key in datasets:
    data = datasets[key]
    times = data.groupby('userid').timestamp.apply(lambda x: x.tolist())
    with open(key+'-time.txt', 'w') as f_time:
        for idx in times.index:
            f_time.write((' '.join(str(x) for x in times[idx])) + '\n')

    m = np.median(data.sentiment)
    events = data.groupby('userid').sentiment.apply(lambda x: [1 if y < m else 2 for y in x])
    with open(key+'-event.txt', 'w') as f_event:
        for idx in events.index:
            f_event.write((' '.join(str(x) for x in events[idx])) + '\n')




