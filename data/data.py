from datetime import datetime, timedelta
import pymongo
import pandas as pd
import numpy as np
import calendar



def get_weekend(row):
    day = row['day']
    if day == 'Sunday' or day == 'Saturday':
        return 'weekend'
    else:
        return 'weekday'


def get_weekday(row):
    return calendar.day_name[row['Time Stamp'].weekday()]


def edit_weather(row):
    weather = row['conditions']
    if isinstance(weather, str):
        if 'Snow' in weather:
            return 'Snow, Rain, Overcast'
        elif 'Rain' in weather:
            return 'Snow, Rain, Overcast'
        elif 'Overcast' in weather:
            return 'Snow, Rain, Overcast'
        else:
            return 'Clear, Partially Cloudy'
    else:
        return weather


def get_hour_db(row):
    hour = row['X']
    time = datetime.strptime(hour, '%H:%M:%S')
    if time.minute >= 30:
        new_time = time + timedelta(hours=1)
        time = time.replace(second=0, minute=0, hour=new_time.hour)
    else:
        time = time.replace(second=0, minute=0)
    return time.hour


def get_from_db():
    # connect to db
    client = pymongo.MongoClient(
        "INSERT MONGODB CONNECTION LINK")

    # get all data
    db = client["DB"]
    collection = db["COLLECTION"]
    df = pd.DataFrame.from_records(collection.find())

    # edit dataframe
    df.columns = ['_id', 'coords', 'title', 'id', 'summary', 'location', 'location_desc', 'road_desc',
                  'section', 'direction', 'direction_desc', 'lane', 'velocity_limit', 'date',
                  'X', 'Y', 'velocity', 'gap', 'occ', 'desc']

    # edit hour
    df['X'] = df.apply(lambda row: get_hour_db(row), axis=1)

    # get weather
    # get_weather(df,"2020")
    weather = pd.read_csv(r'weather2020.csv')
    weather = weather[['Date time', 'Temperature', 'Relative Humidity', 'Cloud Cover', 'Conditions']]
    weather.columns = ['Date time', 'temperature', 'humidity', 'cloud_cover', 'conditions']

    # merge dataframes by date and hour
    weather['Time Stamp'] = pd.to_datetime(weather['Date time'])
    df['Time Stamp'] = pd.to_datetime(df['date'])
    weather = weather.sort_values('Time Stamp')
    traffic = df.sort_values('Time Stamp')
    df = pd.merge(traffic, weather, on='Time Stamp', how='outer')
    df = df.dropna(subset=['X', 'Y'])

    # edit days
    df['day'] = df.apply(lambda row: get_weekday(row), axis=1)
    df['weekend'] = df.apply(lambda row: get_weekend(row), axis=1)

    df = df[
        ['date', 'X', 'Y', 'title', 'coords', 'id', 'summary', 'location', 'location_desc', 'road_desc',
         'section', 'direction', 'direction_desc', 'lane', 'velocity_limit', 'velocity', 'gap', 'occ', 'desc',
         'temperature', 'humidity', 'cloud_cover', 'conditions', 'day', 'weekend']]
    df['conditions'] = df.apply(lambda row: edit_weather(row), axis=1)

    # remove entries without data
    df = df[~df['desc'].str.contains('Ni podatka')]
    df.to_csv(r'.\db_2020_vse.csv', index=False)


def get_date_hour(row):
    date = row['date']
    if (type(date) is str):
        month = int(date[0:2])
        day = int(date[3:5])
        year = int(date[6:10])
        hour = int(date[11:13])
        return datetime(year, month, day, hour), hour
    else:
        return np.NaN, np.NaN


def format_count(row):
    count = row['Y']
    if (type(count) is str):
        if len(count) > 3:
            return int(count[0:1] + count[2:5])
        else:
            return int(count)
    else:
        return int(count)


def get_from_disk():
    data = ['disk_2019_J', 'disk_2020_J']

    for file in data:
        df = pd.read_csv(r'.\/' + file + '.csv', encoding='unicode_escape')
        df['personal'] = df['Motor-MO'] + df['Motor-MO.1'] + df['Osebni-OA'] + df['Osebni-OA.1']
        df['bus'] = df['BUS'] + df['BUS.1']
        df['freight'] = df['La.Tov-LT'] + df['La.Tov-LT.1'] + df['Sr.Tov-ST'] + df['Sr.Tov-ST.1'] + df[
            'Te.Tov-TT'] + df['Te.Tov-TT.1'] + df['T.s Pr-TP'] + df['T.s Pr-TP.1'] + df['Vla?-TPP'] + \
                        df['Vla?-TPP.1']
        df = df[['DD.MM.LL HH:MM', 'SUMA1', 'SUMA2', 'SUMA3', 'personal', 'bus', 'freight']]
        df.columns = ['date', 'sum1', 'sum2', 'Y','personal','bus','freight']
        df = df.dropna()
        df['X'] = 0
        df[['date', 'X']] = df.apply(lambda row: get_date_hour(row), axis=1, result_type="expand")
        df = df.dropna()
        df = df[['date', 'X', 'sum1', 'sum2', 'Y','personal','bus','freight']]

        df['Y'] = df.apply(lambda row: format_count(row), axis=1)

        # get_weather(df,file)
        if "2019" in file:
            weather = pd.read_csv(r'weather2019.csv')
        else:
            weather = pd.read_csv(r'weather2020.csv')
        weather = weather[['Date time', 'Temperature', 'Relative Humidity', 'Cloud Cover', 'Conditions']]
        weather.columns = ['Date time', 'temperature', 'humidity', 'cloud_cover', 'conditions']

        # merge dataframes by date and hour
        weather['Time Stamp'] = pd.to_datetime(weather['Date time'])
        df['Time Stamp'] = pd.to_datetime(df['date'])
        weather = weather.sort_values('Time Stamp')
        traffic = df.sort_values('Time Stamp')
        df = pd.merge(traffic, weather, on='Time Stamp', how='outer')
        df = df.dropna(subset=['X', 'Y'])

        # edit days
        df['day'] = df.apply(lambda row: get_weekday(row), axis=1)
        df['weekend'] = df.apply(lambda row: get_weekend(row), axis=1)

        df = df[['date', 'X', 'sum1', 'sum2', 'Y', 'day', 'weekend','personal','bus','freight','conditions']]

        df['conditions'] = df.apply(lambda row: edit_weather(row), axis=1)
        df.to_csv(r'.\/' + file + "_df.csv", index=False)


"""
    ###
    # Main
    ###
"""
get_from_db()
get_from_disk()
