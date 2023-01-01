import pandas as pd
import numpy as np
import scipy.io as sio
from datetime import datetime

TF_PATH = './data/track_features/tf_mini.csv'
TRAINING_PATH = './data/training_set/log_mini.csv'
Y_PATH = 'data/preprocessed_data/y.csv'
X_PATH = 'data/preprocessed_data/X.csv'

CURRENT_TRACK_COLUMNS = [ \
    'duration', \
    'release_year', \
    'us_popularity_estimate', \
    'acousticness', \
    'beat_strength', \
    'bounciness', \
    'danceability', \
    'dyn_range_mean', \
    'energy', \
    'flatness', \
    'instrumentalness', \
    'key', \
    'liveness', \
    'loudness', \
    'mechanism', \
    'mode', \
    'organism', \
    'speechiness', \
    'tempo', \
    'time_signature', \
    'valence' \
]

SESSION_AVERAGE_COLUMNS = [ \
    'duration', \
    'release_year', \
    'us_popularity_estimate', \
    'acousticness', \
    'beat_strength', \
    'bounciness', \
    'danceability', \
    'dyn_range_mean', \
    'energy', \
    'flatness', \
    'instrumentalness', \
    'key', \
    'liveness', \
    'loudness', \
    'mechanism', \
    'mode', \
    'organism', \
    'speechiness', \
    'tempo', \
    'time_signature', \
    'valence' \
]

def datestr_to_day(date_string):
    """Returns the day from a string of format 'year-month-day'"""

    date = datetime.strptime(date_string, '%Y-%m-%d')
    return date.weekday()

def session_to_result(session):
    """ Returns the ground-truth for a session: The skip_2 value of the last track """

    return session['skip_2'].iloc[-1]


def entry_to_track(session_entry, tf_data):
    """ Returns the track corresponding to an entry in the session data """

    track_id = session_entry['track_id_clean']
    return tf_data.loc[track_id]


def session_to_data(session, tf_data):
    """Construct a single data entry from a session"""

    session_id = session.iloc[0]['session_id']
    result = pd.DataFrame(index=[session_id])
    # result['session_id'] = [session.iloc[0]['session_id']]

    # get the neccessary columns from the current track (the last one) 
    # and put this into the resulting dataframe
    track_data = entry_to_track(session.iloc[-1], tf_data)[CURRENT_TRACK_COLUMNS]

    for column in CURRENT_TRACK_COLUMNS:
        result[f'track_{column}'] = [track_data[column]]

    previous_session = session.iloc[:-1]

    # find the corresponding tracks from the previous session and find the averages
    previous_tracks = previous_session.apply(lambda s: entry_to_track(s, tf_data), axis=1)
    session_averages = previous_tracks[SESSION_AVERAGE_COLUMNS].apply(np.mean, axis=0)

    # put these averages into the resulting dataframe
    for column in SESSION_AVERAGE_COLUMNS:
        result[f'ses_{column}'] = [session_averages[column]]

    # add the average mean and std from the skip 
    result['skip_mean'] = [previous_session['skip_2'].apply(np.mean)]
    result['skip_std'] = [previous_session['skip_2'].apply(np.std)]

    # get the context type and premium from the session
    result['premium'] = previous_session.iloc[0]['premium']
    result['context_type'] = previous_session.iloc[0]['context_type']

    # get the day from the session
    result['day'] = [datestr_to_day(previous_session['date'].iloc[0])]

    # print(result)
    return result


def load_track_features(path):
    """ loads the track features and preprocesses them """

    tf_data = pd.read_csv(path)
    tf_data.set_index(tf_data['track_id'], inplace=True)
    tf_data.drop('track_id', axis=1, inplace=True)
    tf_data['mode'] = tf_data['mode'].apply(lambda m: 0 if m == 'minor' else 0)

    return tf_data


def main():
    """ Main function """
    tf_data = load_track_features(TF_PATH)
    session_data = pd.read_csv(TRAINING_PATH)

    # group the data by session
    groups = session_data.groupby(['session_id'], group_keys=False)

    # write the data of the session to the x file
    print(f'writing data to {X_PATH}')
    X_data = groups.apply(lambda s: session_to_data(s, tf_data))
    print(X_data.head(10))
    X_data.to_csv(X_PATH)

    # write the result of the session to the y file
    print(f'writing results to {Y_PATH}')
    groups.apply(session_to_result).to_csv(Y_PATH)

    

    print('succesfully preprocessed data')


if __name__=="__main__":
    main()


