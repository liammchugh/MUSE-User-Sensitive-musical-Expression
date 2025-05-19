''' 
Sample Code from Tyler Hutcherson https://github.com/tylerhutcherson
2025: Skafos is outdated and no longer available
This code is a sample of how to clean and prepare the data for training an activity classifier
'''

import json
from datetime import datetime, timedelta
import s3fs
import pandas as pd

# Connect to s3 bucket
s3 = s3fs.S3FileSystem(anon=False)
s3_url = 's3://skafos.example.data/ActivityClassifier/raw'

# Get file listing from s3 bucket
all_files = s3.ls(s3_url)
watch_files = [file for file in all_files if 'watch' in file]
session_file = [file for file in all_files if 'sessions' in file][0]

# Load session activity log from s3
with s3.open(session_file, 'r') as f:
    session_log = json.loads(f.read())
    
### Constants and Functions ###
# Two rounds of data collection
rounds = ['round1', 'round2']
# For each data collection round there were 2 users participating
users_per_round = 2
# For each data collection round there were 3 separate sessions
sessons_per_round = 3

# Timestamp format to follow
time_format = '%Y-%m-%d %H:%M:%S.%f'

# Columns to use from the csv data
session_sensor_data_columns = [
    "loggingTime(txt)",
    "motionRotationRateX(rad/s)",
    "motionRotationRateY(rad/s)",
    "motionRotationRateZ(rad/s)",
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)"
]

# Helper functions
def convert_timestamp(x):
    return datetime.fromtimestamp(x).strftime(time_format)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def make_delta(time):
    h, m, seconds = time.split(':')
    s, milli = seconds.split('.')
    milli = milli + str(0)
    return timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(milli))

################################

# Load Activity Data

# Iterate through all user session logs
# Map to the proper activity label
# Concatenate into a single activity dataframe

session_id = 0 # Keep track of session id: unique to each user file
activity_data = pd.DataFrame()

# For each round of data collection
for rnd in rounds:
    print(f'Parsing {rnd} activity data', flush=True)
    # Grab the files for this round
    rnd_files = [file for file in watch_files if rnd in file]
    # Sort files for this round by timestamp
    rnd_files = sorted(
        rnd_files,
        key=lambda x: datetime.strptime(x.split('__')[2][:19], '%Y-%m-%d_%H-%M-%S'),
        reverse=False
    )
    # Group files for the round by session
    rnd_session_files = list(chunks(rnd_files, users_per_round))
    # Should have the right number of sessions, each file within a session is a unique user
    assert len(rnd_session_files) == sessons_per_round
    # For each session (3) within the round
    for session in range(sessons_per_round):
        print(f'Parsing session {session + 1} data files', flush=True)
        # Grab the user activity log files for this session
        session_files = rnd_session_files[session]
        # Grab the activity labels for this session
        session_log_data = session_log[rnd]['session' + str(1+session)]
        # For each user
        for user_file in session_files:
            # Load user file for this session - fix timestamp and add session id
            user_log_df = pd.read_csv('s3://' + user_file).reset_index()[session_sensor_data_columns]
            user_log_df["loggingTime"] = user_log_df["loggingTime(txt)"].apply(lambda x: pd.to_datetime(x))
            user_log_df.drop("loggingTime(txt)", axis=1, inplace=True)
            user_log_df["sessionId"] = session_id
            session_id += 1
            # Convert timestamp and make sure it's ordered appropriately
            user_log_df.sort_values(by="loggingTime", ascending=True)
            first_val = user_log_df["loggingTime"][0]
            # Get the logs that contain the activity labels for this session
            user_session_activity_df = pd.DataFrame({
                'activity': pd.Series([s[0] for s in session_log_data], dtype=str),
                'loggingTime': pd.Series([first_val + make_delta(s[1]) for s in session_log_data])
            }).sort_values(by="loggingTime", ascending=True)
            # Fuzzy merge on timestamps to map user logs to activity labels
            user_log_cleaned = pd.merge_asof(
                left=user_log_df,
                right=user_session_activity_df,
                on='loggingTime',
                direction='forward'
            )
            activity_data = pd.concat((activity_data, user_log_cleaned))
            
# In the end, you will have the fully cleaned and joined activity data file
# From here we can train a model on Skafos with this data (https://github.com/skafos/TuriActivityClassifier)
activity_data.head()