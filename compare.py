import data_processing as dproc
import helpers as hlp
import pandas as pd


def compare_by_day(df, n_components, model_type):
    slo_names = dict({
        "Monday": "ponedeljek",
        "Tuesday": "torek",
        "Wednesday": "sreda",
        "Thursday": "četrtek",
        "Friday": "petek",
        "Saturday": "sobota",
        "Sunday": "nedelja"
    })

    dproc.compare_by_component(df, 'day', n_components, model_type, [0, 0, 0, 0, 1, 1, 0], ['delovnik', 'vikend'], [False, False, False, True, True, True, False], rows=2, cols=1,
                               labels=slo_names,save_name='dnevi')


def compare_by_weekend(df,n_components,model_type):
    slo_names = dict({
        "weekday": "delovnik",
        "weekend": "vikend",
    })

    dproc.compare_by_component(df, 'weekend', n_components, model_type,[0,0],[''],[True,True],labels=slo_names,save_name='vikendi')

def compare_by_type(df,n_components,model_type):
    slo_names = dict({
        "personal": "osebna",
        "bus": "avtobusi",
        "freight": "tovorna",

    })

    dproc.compare_by_component(df,['personal', 'bus', 'freight'], n_components, model_type,[0,1,0],['osebna, tovorna vozila','avtobusi'],[True,True,True],labels=slo_names,save_name='tipi_vozil',rows=2, cols=1,multiple_cols=True)

def compare_by_weather(df,n_components,model_type):
    slo_names = dict({
        "Snow, Rain, Overcast": "dež, sneg, megla",
        "Clear, Partially Cloudy": "jasno, rahla oblačnost"
    })
    df = df.dropna(subset=['conditions'])
    dproc.compare_by_component(df, 'conditions', n_components, model_type,[0,0],[''],[True,True],labels=slo_names,save_name='vreme')

def compare_by_covid(n_components,model_type):

    df = pd.read_csv(r'data\disk_2019_J_df.csv')

    april_2019 = df[df['date'].str.contains('2019-04-')].copy()
    april_2019['year'] = "april 2019"

    april_2020 = pd.read_csv(r'data\disk_2020_J_df.csv')
    april_2020 = april_2020[april_2020['date'].str.contains('2020-04-')]
    april_2020['year'] = "april 2020"

    november_2019 = df[df['date'].str.contains('2019-11-')].copy()
    november_2019['year'] = "november 2019"

    november_2020 = pd.read_csv(r'data/db_2020_vse.csv')
    df_direction1 = november_2020[november_2020['id'].str.contains('0178-11')]
    df_direction2 = november_2020[november_2020['id'].str.contains('0178-21')]
    df_direction3 = november_2020[november_2020['id'].str.contains('0178-12')]
    november_2020 = pd.merge(df_direction1, df_direction2, how='inner', on=['date', 'X'])
    november_2020 = pd.merge(november_2020, df_direction3, how='inner', on=['date', 'X'])
    november_2020['Y'] = november_2020['Y_x'] + november_2020['Y_y'] + november_2020['Y']
    november_2020['year'] = "november 2020"

    # ALL DAYS
    df_1=hlp.clean_data(april_2019)
    df_2=hlp.clean_data(april_2020)
    df_3=hlp.clean_data(november_2019)
    df_4=hlp.clean_data(november_2020)
    df_together = pd.concat([df_1, df_2, df_3, df_4])
    dproc.compare_by_component(df_together, 'year', n_components, model_type,[0,0,1,1],['april','november'],[True,True,True,True],rows=1, cols=2,save_name='covid')

    # ONLY WEEKENDS
    df_1 = april_2019[april_2019['weekend'].str.contains('weekend')]
    df_2 = april_2020[april_2020['weekend'].str.contains('weekend')]
    df_3 = november_2019[november_2019['weekend'].str.contains('weekend')]
    df_4 = november_2020[november_2020['weekend'].str.contains('weekend')]
    df_1 = hlp.clean_data(df_1)
    df_2 = hlp.clean_data(df_2)
    df_3 = hlp.clean_data(df_3)
    df_4 = hlp.clean_data(df_4)
    df_together = pd.concat([df_1, df_2, df_3, df_4])
    dproc.compare_by_component(df_together, 'year', n_components, model_type,[0,0,1,1],['vikendi - april','vikendi - november'],[True,True,True,True],rows=1, cols=2,save_name='covid_vikend')


    # ONLY WORK DAYS
    df_1 = april_2019[april_2019['weekend'].str.contains('weekday')]
    df_2 = april_2020[april_2020['weekend'].str.contains('weekday')]
    df_3 = november_2019[november_2019['weekend'].str.contains('weekday')]
    df_4 = november_2020[november_2020['weekend'].str.contains('weekday')]
    df_1 = hlp.clean_data(df_1)
    df_2 = hlp.clean_data(df_2)
    df_3 = hlp.clean_data(df_3)
    df_4 = hlp.clean_data(df_4)
    df_together = pd.concat([df_1, df_2, df_3, df_4])
    dproc.compare_by_component(df_together, 'year', n_components, model_type, [0, 0, 1, 1],
                               ['delovnik - april', 'delovnik - november'], [True, True, True, True], rows=1, cols=2,
                               save_name='covid_delovnik')