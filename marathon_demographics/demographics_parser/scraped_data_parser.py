import pandas as pd
from datetime import datetime, timedelta


chicago2017 = pd.read_csv('./chicago2017.csv').dropna()
chicago2017 = chicago2017[chicago2017['half_times'] != '–']

chicago2016 = pd.read_csv('./chicago2016.csv').dropna()
chicago2016 = chicago2016[chicago2016['half_times'] != '–']
chicago2016 = chicago2016[chicago2016['ages'] != '–']

chicago2015 = pd.read_csv('./chicago2015.csv').dropna()
chicago2015 = chicago2015[chicago2015['half_times'] != '–']

chicago2014 = pd.read_csv('./chicago2014.csv').dropna()
chicago2014 = chicago2014[chicago2014['half_times'] != '–']
chicago2014 = chicago2014[chicago2014['ages'] != 'W-15']
chicago2014 = chicago2014[chicago2014['ages'] != 'M-15']

london2017 = pd.read_csv('./london2017.csv').dropna()
london2016 = pd.read_csv('./london2016.csv').dropna()
london2015 = pd.read_csv('./london2015.csv').dropna()
london2015 = london2015[london2015['half_times'] != 'DSQ']

london2014 = pd.read_csv('./london2014.csv').dropna()


def pace(df):
    df['finish_secs'] = pd.to_timedelta(df['finish_times'])
    df['half_secs'] = pd.to_timedelta(df['half_times'])
    df['pace'] = (df['finish_secs'].dt.total_seconds() 
    - df['half_secs'].dt.total_seconds())/ df['half_secs'].dt.total_seconds()
    
    df = df.sort_values(['finish_secs'], ascending=[True])
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.reset_index(drop=True)
    
    return df

def gender(df):
    rankings,gender = [], []
    for i,ranking in enumerate(df['gender_rankings']):
        if ranking not in rankings:
            rankings.append(ranking)
            gender.append('M')
        else:
            gender.append('F')

    df['gender'] = gender

    return df


#boston qualifier analysis london
time_men_london = {'18-39':datetime.strptime('03:05:00', '%H:%M:%S'),
            '40-44':datetime.strptime('03:10:00', '%H:%M:%S'),
            '45-49':datetime.strptime('03:20:00', '%H:%M:%S'),
            '50-54':datetime.strptime('03:25:00', '%H:%M:%S'),
            '55-59':datetime.strptime('03:35:00', '%H:%M:%S'),
            '60-64':datetime.strptime('03:50:00', '%H:%M:%S'),
            '65-69':datetime.strptime('04:05:00', '%H:%M:%S'),
            '70+':datetime.strptime('04:20:00', '%H:%M:%S')
            }

time_women_london = {'18-39':datetime.strptime('03:35:00', '%H:%M:%S'),
              '40-44':datetime.strptime('03:40:00', '%H:%M:%S'),
              '45-49':datetime.strptime('03:50:00', '%H:%M:%S'),
              '50-54':datetime.strptime('03:55:00', '%H:%M:%S'),
              '55-59':datetime.strptime('04:05:00', '%H:%M:%S'),
              '60-64':datetime.strptime('04:20:00', '%H:%M:%S'),
              '65-69':datetime.strptime('04:35:00', '%H:%M:%S'),
              '70+':datetime.strptime('04:50:00', '%H:%M:%S')
              }

#boston qualifier analysis chicago
time_men_chicago = {'16-19':datetime.strptime('03:00:00', '%H:%M:%S'),
            '20-24':datetime.strptime('03:00:00', '%H:%M:%S'),
            '25-29':datetime.strptime('03:00:00', '%H:%M:%S'), 
            '30-34':datetime.strptime('03:00:00', '%H:%M:%S'),
            '35-39':datetime.strptime('03:05:00', '%H:%M:%S'),
            '40-44':datetime.strptime('03:10:00', '%H:%M:%S'),
            '45-49':datetime.strptime('03:20:00', '%H:%M:%S'),
            '50-54':datetime.strptime('03:25:00', '%H:%M:%S'),
            '55-59':datetime.strptime('03:35:00', '%H:%M:%S'),
            '60-64':datetime.strptime('03:50:00', '%H:%M:%S'),
            '65-69':datetime.strptime('04:05:00', '%H:%M:%S'),
            '70-74':datetime.strptime('04:20:00', '%H:%M:%S'),
            '75-79':datetime.strptime('04:35:00', '%H:%M:%S'),
            '80+':datetime.strptime('04:50:00', '%H:%M:%S')
            }

time_women_chicago = {'16-19':datetime.strptime('03:30:00', '%H:%M:%S'),
              '20-24':datetime.strptime('03:30:00', '%H:%M:%S'),
              '25-29':datetime.strptime('03:30:00', '%H:%M:%S'), 
              '30-34':datetime.strptime('03:30:00', '%H:%M:%S'),
              '35-39':datetime.strptime('03:35:00', '%H:%M:%S'),
              '40-44':datetime.strptime('03:40:00', '%H:%M:%S'),
              '45-49':datetime.strptime('03:50:00', '%H:%M:%S'),
              '50-54':datetime.strptime('03:55:00', '%H:%M:%S'),
              '55-59':datetime.strptime('04:05:00', '%H:%M:%S'),
              '60-64':datetime.strptime('04:20:00', '%H:%M:%S'),
              '65-69':datetime.strptime('04:35:00', '%H:%M:%S'),
              '70-74':datetime.strptime('04:50:00', '%H:%M:%S'),
              '75-79':datetime.strptime('05:05:00', '%H:%M:%S'),
              '80+':datetime.strptime('05:20:00', '%H:%M:%S')
              }

def bq(df, table_men, table_women):
    BQ = []
    for i in range(len(df['finish_times'])):
        if df['gender'][i] == 'M':
            if df['finish_secs'][i] < timedelta(hours = table_men[df['ages'][i]].hour, minutes = table_men[df['ages'][i]].minute, seconds = table_men[df['ages'][i]].second):
                BQ.append('YES')
            else: BQ.append('NO')
        elif df['gender'][i] == 'F':
            if df['finish_secs'][i] < timedelta(hours = table_women[df['ages'][i]].hour, minutes = table_women[df['ages'][i]].minute, seconds = table_women[df['ages'][i]].second):
                BQ.append('YES')
            else: BQ.append('NO')

    df['BQ'] = BQ   
    
    return df         

#chicago2014 = bq(gender(pace(chicago2014)), time_men_chicago, time_women_chicago)
#chicago2014.to_csv('./chicago2014_clean.csv')
#chicago2015 = bq(gender(pace(chicago2015)), time_men_chicago, time_women_chicago)
#chicago2015.to_csv('./chicago2015_clean.csv')
#chicago2016 = bq(gender(pace(chicago2016)), time_men_chicago, time_women_chicago)
#chicago2016.to_csv('./chicago2016_clean.csv')
#chicago2017 = bq(gender(pace(chicago2017)), time_men_chicago, time_women_chicago)
#chicago2017.to_csv('./chicago2017_clean.csv')

#london2014 = bq(gender(pace(london2014)), time_men_london, time_women_london)
#london2014.to_csv('./london2014_clean.csv')
#london2015 = bq(gender(pace(london2015)), time_men_london, time_women_london)
#london2015.to_csv('./london2015_clean.csv')
#london2016 = bq(gender(pace(london2016)), time_men_london, time_women_london)
#london2016.to_csv('./london2016_clean.csv')
#london2017 = bq(gender(pace(london2017)), time_men_london, time_women_london)
#london2017.to_csv('./london2017_clean.csv')