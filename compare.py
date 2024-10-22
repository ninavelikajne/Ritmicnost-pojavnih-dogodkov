import data_processing as dproc
import pandas as pd

models_type = ['poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb']


def compare_by_day():
    df = pd.read_csv(r'data\J_2019.csv')
    df = dproc.clean_data(df)
    df_results = dproc.compare_by_component(df, 'dan', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], models_type, [0, 0, 0, 0, 1, 1, 0],
                                            ['delovnik', 'vikend'], rows=2, cols=1, save_file_to='dnevi.pdf')

    return df_results


def compare_by_weekend():
    df = pd.read_csv(r'data\J_2019.csv')
    df = dproc.clean_data(df)
    df_results = dproc.compare_by_component(df, 'vikend_delovnik', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], models_type, [0, 0], [''], save_file_to='vikendi.pdf')
    return df_results


def compare_by_type():
    slo_names = dict({
        "osebno_v": "osebna v.",
        "avtobus": "avtobusi",
        "tovorno_v": "tovorna v.",

    })
    df = pd.read_csv(r'data\J_2019.csv')
    df = dproc.clean_data(df)
    df_results = dproc.compare_by_component(df, ['osebno_v', 'avtobus', 'tovorno_v'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                            models_type, [0, 1, 0], ['osebna in tovorna vozila', 'avtobusi'],
                                            labels=slo_names, save_file_to='tipi_vozil.pdf', rows=2, cols=1,
                                            multiple_cols=True, main_name='tipi_vozil')
    return df_results


def compare_by_weather():
    df = pd.read_csv(r'data\J_2019.csv')
    df = dproc.clean_data(df)
    df = df.dropna(subset=['vreme'])
    df_results = dproc.compare_by_component(df, 'vreme', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], models_type, [0, 0], [''], save_file_to='vreme.pdf')
    return df_results


def compare_by_covid():
    file_names = ['covid_all', 'covid_work', 'covid_week']
    results = []
    for file_name in file_names:
        df = pd.read_csv(r'data\/' + file_name + '.csv')
        df_results = dproc.compare_by_component(df, 'obdobje', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], models_type, [0, 1, 0, 1],
                                                ['april', 'november'], rows=1, cols=2, save_file_to=file_name + '.pdf')
        results.append(df_results)

    return results
