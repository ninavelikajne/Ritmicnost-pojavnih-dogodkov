import pandas as pd
from matplotlib import MatplotlibDeprecationWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import data_processing as dproc
import compare as comp
import helpers as hlp
import warnings


# ignore these warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

file_names = ['disk_2019_J_df', 'disk_2019_smartinska_df']
title_names=['ljubljanska juzna obvoznica','Šmartinska cesta']

hlp.plot_raw_data(file_names,title_names,'Število vozil',[1500,200])

for file_name in file_names:

    df = pd.read_csv(r'data\/' + file_name+'.csv')
    models_type = ['poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb']

    if file_name == 'disk_2019_smartinska_df':
        n_components = [1, 2, 3, 4, 5, 6]
        save_file_to = 'smartinska'
    else:
        n_components = [1, 2, 3, 4, 5, 6, 7, 8]
        save_file_to = 'juzna'

    # clean data
    df = hlp.clean_data(df)

    # find the best model - fit to models
    df_best = dproc.find_best_model(df,models_type,n_components,save_file_to=save_file_to)
    print(df_best.model_type, df_best.n_components)
    model_type = df_best.model_type
    n_components = df_best.n_components

    if file_name == 'disk_2019_J_df':
        model_type1=model_type
        n_components1=n_components

        # compare
        comp.compare_by_day(df,n_components,model_type)
        comp.compare_by_weekend(df,n_components,model_type)
        comp.compare_by_weather(df,n_components,model_type)
        comp.compare_by_type(df,n_components,model_type)
        comp.compare_by_covid(n_components,model_type)
    else:
        model_type2=model_type
        n_components2=n_components

# plot best models and CIs
df1 = pd.read_csv(r'data\disk_2019_J_df.csv')
df1 = hlp.clean_data(df1)
df2 = pd.read_csv(r'data\disk_2019_smartinska_df.csv')
df2 = hlp.clean_data(df2)
hlp.plot_models([df1, df2], [model_type1, model_type2], [n_components1, n_components2], title=['ljubljanska južna obvoznica', 'Šmartinska cesta'], cols=1, rows=2)
hlp.plot_CIs([df1, df2], [model_type1, model_type2], [n_components1, n_components2], title=['ljubljanska južna obvoznica', 'Šmartinska cesta'], cols=1, rows=2)
