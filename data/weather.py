import requests
import datetime


def fetch_weather(start_date, end_date, cnt, year):
    url = "https://rapidapi.p.rapidapi.com/history"
    querystring = {"startDateTime": start_date, "aggregateHours": "1", "location": "Ljubljana", "endDateTime": end_date,
                   "unitGroup": "uk", "dayStartTime": "00:00:00", "contentType": "csv", "dayEndTime": "23:00:00",
                   "shortColumnNames": "0"}
    headers = {
        'x-rapidapi-host': "visual-crossing-weather.p.rapidapi.com",
        'x-rapidapi-key': "INSERT API KEY"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    result = response.text

    if "2019" in year:
        file_name = "weather2019.csv"
    else:
        file_name = "weather2020.csv"

    if cnt != 0:
        result = '\n'.join(response.text.split('\n')[1:])
        f = open(file_name, "a")
    else:
        f = open(file_name, "w+")
    f.write(result)
    f.close()


def get_weather(df, year):
    if "disk_2020" in year:
        first_date = datetime.datetime(2020, 1, 1, 0)
        last_date = datetime.datetime(2020, 11, 3, 23)
    elif "disk_" in year:
        first_date = datetime.datetime.strptime(df.iloc[0]['date'], "%Y-%m-%d %H:%M:%S")
        last_date = datetime.datetime.strptime(df.iloc[-1]['date'], "%Y-%m-%d %H:%M:%S")
    else:  # db format
        first_date = datetime.datetime.strptime(df.iloc[0]['date'], "%d/%m/%Y")
        last_date = datetime.datetime.strptime(df.iloc[-1]['date'], "%d/%m/%Y")
        last_date = last_date + datetime.timedelta(hours=23)

    start_temp_date = first_date
    end_temp_date = first_date + datetime.timedelta(days=14, hours=23)
    cnt = 0
    while not (start_temp_date > last_date):
        start = start_temp_date.strftime("%Y-%m-%dT%H:%M:%S")
        end = end_temp_date.strftime("%Y-%m-%dT%H:%M:%S")
        fetch_weather(start, end, cnt, year)

        start_temp_date = end_temp_date + datetime.timedelta(hours=1)
        end_temp_date = start_temp_date + datetime.timedelta(days=14, hours=23)
        cnt = cnt + 1
