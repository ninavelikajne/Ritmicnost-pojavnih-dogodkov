import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from collections import OrderedDict
import matplotlib.dates as mdates
import matplotlib.dates as md
import math
import copy
import random
import helpers as hlp

colors = ['blue', 'green', 'orange', 'red', 'purple', 'olive', 'tomato', 'yellow', 'pink', 'turquoise', 'lightgreen']


def plot_models(dfs, model_type, n_components, title=[''], rows=1, cols=1, plot_CIs=True, repetitions=20,
                save_file_to='zmagovalni.pdf'):
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    i = 0
    for df in dfs:
        results, stats, X_test, Y_test, _ = fit_to_model(df, n_components[i], model_type[i])

        # plot
        ax = fig.add_subplot(gs[i])
        if plot_CIs:
            subplot_confidence_intervals(df, n_components[i], model_type[i], ax, repetitions=repetitions)

        subplot_model(df['X'], df['Y'], X_test, Y_test, ax, color='blue', title=title[i], label='prilagojena krivulja')

        i = i + 1

    ax_list = fig.axes
    for ax in ax_list:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='large')

    fig.tight_layout()
    fig.savefig(r'results\/' + save_file_to)
    plt.show()


def plot_raw_data(file_names, title, hour_intervals, cols=1, rows=2, save_file_to='izvorni.pdf'):
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    ix = 0
    for file_name in file_names:
        df = pd.read_csv(r'.\/data\/' + file_name + ".csv")
        df = clean_data(df)

        var = df[['Y']].to_numpy().var()
        mean = df[['Y']].to_numpy().mean()
        print(file_name, ": Var: ", var, " Mean: ", mean)

        ax = fig.add_subplot(gs[ix])
        ax.scatter(df.date.head(500), df.Y.head(500), c='blue', s=1)

        date_form = md.DateFormatter("%d-%m %H:00")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_intervals[ix]))
        plt.xticks(rotation=45)
        plt.xlabel('Čas [D-M H:MIN]')
        plt.ylabel('Število')
        plt.title(title[ix])

        ix = ix + 1

    fig.tight_layout()
    fig.savefig(r'results\/' + save_file_to)
    plt.show()


def clean_data(df):
    df = df.dropna(subset=['X', 'Y'])

    for hour in range(0, 24, 1):
        df_hour = df.loc[df.X == hour].copy()
        # cleaning outliers
        df_hour = df_hour.loc[df_hour.Y >= df_hour.Y.quantile(0.15)].copy()
        df_hour = df_hour.loc[df_hour.Y <= df_hour.Y.quantile(0.85)].copy()
        df.loc[df['X'] == hour, ['Y']] = df_hour['Y']

    df = df.dropna(subset=['X', 'Y'])
    return df


def find_best_model(df, models_type, n_components, plot=False, save_file_to='modeli.pdf'):
    df_results = pd.DataFrame()

    if plot:
        rows, cols = hlp.get_factors(len(models_type))
        fig = plt.figure(figsize=(8 * cols, 8 * rows))
        gs = gridspec.GridSpec(rows, cols)

    i = 0
    for model_type in models_type:
        c = 0
        for n_component in n_components:
            results, stats, X_test, Y_test, _ = fit_to_model(df, n_component, model_type)

            # plot
            if plot:
                ax = fig.add_subplot(gs[i])
                title = hlp.get_slo_model_name(model_type)
                subplot_model(df['X'], df['Y'], X_test, Y_test, ax, color=colors[c], title=title,
                              label='N=' + str(n_component))

            df_results = df_results.append(stats, ignore_index=True)
            c = c + 1

        i = i + 1

    # show plots
    if plot:
        ax_list = fig.axes
        for ax in ax_list:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='medium')

        fig.tight_layout()
        fig.savefig(r'results\/' + save_file_to)
        plt.show()

    # eval
    best_n_components = get_best_n_components(df_results).n_components
    best_model_type = get_best_model_type(df_results, 'Vuong').model_type
    df_best=df_results.loc[(df_results['n_components']==best_n_components) & (df_results['model_type'].str.match(best_model_type))].copy()

    return df_best.iloc[0]


def cosinor(X, n_components, period=24, lin_comp=False):
    X_test = np.linspace(0, 100, 1000)

    for i in range(n_components):
        k = i + 1
        A = np.sin((X / (period / k)) * np.pi * 2)
        B = np.cos((X / (period / k)) * np.pi * 2)

        A_test = np.sin((X_test / (period / k)) * np.pi * 2)
        B_test = np.cos((X_test / (period / k)) * np.pi * 2)

        if i == 0:
            X_fit = np.column_stack((A, B))
            X_fit_test = np.column_stack((A_test, B_test))
        else:
            X_fit = np.column_stack((X_fit, A, B))
            X_fit_test = np.column_stack((X_fit_test, A_test, B_test))

    X_fit_eval_params = X_fit_test

    if lin_comp and n_components:
        X_fit = np.column_stack((X, X_fit))
        X_fit_eval_params = np.column_stack((np.zeros(len(X_test)), X_fit_test))
        X_fit_test = np.column_stack((X_test, X_fit_test))

    return X_fit, X_test, X_fit_test, X_fit_eval_params


def fit_to_model(df, n_components, model_type):
    X_fit, X_test, X_fit_test, X_fit_eval_params = cosinor(df['X'], n_components=n_components, period=24)
    Y = df['Y'].to_numpy()

    X_fit = sm.add_constant(X_fit, has_constant='add')
    X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

    if model_type == 'poisson':
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(maxiter=5000, maxfun=5000, method='nm', disp=0)
    elif model_type == 'gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(maxiter=5000, maxfun=5000, method='nm', disp=0)
    elif model_type == 'zero_poisson':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(endog=Y, exog=X_fit, exog_infl=X_fit)
        results = model.fit(maxiter=5000, maxfun=5000, method='nm', skip_hessian=True, disp=0)
    elif model_type == 'zero_nb':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(endog=Y, exog=X_fit, exog_infl=X_fit,
                                                                               p=1)
        results = model.fit(maxiter=5000, maxfun=5000, method='nm', skip_hessian=True, disp=0)
    elif model_type == 'nb':
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(maxiter=5000, maxfun=5000, method='nm', disp=0)
    else:
        print("Invalid option")
        return

    if model_type == 'zero_nb' or model_type == "zero_poisson":
        Y_test = results.predict(X_fit_test, exog_infl=X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params, exog_infl=X_fit_eval_params)
        Y_fit = results.predict(X_fit, exog_infl=X_fit)
    else:
        Y_test = results.predict(X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params)
        Y_fit = results.predict(X_fit)

    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
    stats = calculate_statistics(Y, Y_fit, n_components, results, model, model_type, rhythm_params)
    return results, stats, X_test, Y_test, X_fit_test


def calculate_confidence_intervals(df, n_components, model_type, repetitions=20):
    sample_size = round(df.shape[0] - df.shape[0] / 3)
    for i in range(0, repetitions):
        sample = df.sample(sample_size)
        results, _, _, _, _ = fit_to_model(sample, n_components, model_type)
        if i == 0:
            save = pd.DataFrame({str(i): results.params})
        else:
            save[str(i)] = results.params

    columns = save.shape[0]

    mean = save.mean(axis=1)
    std = save.std(axis=1)
    save = pd.DataFrame({"mean": mean, "std": std})
    save['CI1'] = save['mean'] - 1.96 * save['std']
    save['CI2'] = save['mean'] + 1.96 * save['std']

    CIs = pd.DataFrame({0: [], 1: []})
    for i in range(columns):
        CIs = CIs.append({0: save['CI1'].iloc[i], 1: save['CI2'].iloc[i]}, ignore_index=True)

    return CIs


def subplot_model(X, Y, X_test, Y_test, ax, plot_measurements_with_color=False, plot_model=True, title='',
                  raw_data_label='', color='black', label=''):
    ax.set_title(title)
    ax.set_xlabel('Čas [h]')
    ax.set_ylabel('Število')

    if plot_measurements_with_color:
        raw_label = 'izvorni podatki \n- ' + label
        ax.plot(X, Y, 'ko', markersize=1, color=color, label=raw_label)
    else:
        if raw_data_label:
            ax.plot(X, Y, 'ko', markersize=1, color='black', label='izvorni podatki \n- ' + raw_data_label)
        else:
            ax.plot(X, Y, 'ko', markersize=1, color='black', label='izvorni podatki')

    if plot_model:
        ax.plot(X_test, Y_test, 'k', label=label, color=color)

    ax.set_xlim(0, 23)

    return ax


def evaluate_rhythm_params(X, Y, period=24):
    X = X[:period * 10]
    Y = Y[:period * 10]
    m = min(Y)
    M = max(Y)
    A = M - m
    MESOR = m + A / 2
    AMPLITUDE = A / 2

    locs, heights = signal.find_peaks(Y, height=M * 0.75)
    heights = heights['peak_heights']
    x = np.take(X, locs)

    result = {'amplitude': round(AMPLITUDE, 2), 'mesor': round(MESOR, 2), 'locs': np.around(x, decimals=2),
              'heights': np.around(heights, decimals=2)}
    return result


def calculate_statistics(Y, Y_fit, n_components, results, model, model_type, rhythm_param):
    # RSS
    RSS = sum((Y - Y_fit) ** 2)

    # p
    p = results.llr_pvalue

    # AIC
    aic = results.aic

    # BIC
    bic = results.bic

    # llf for each observation
    logs = model.loglikeobs(results.params)

    return {'model_type': model_type, 'n_components': n_components,
            'amplitude': rhythm_param['amplitude'],
            'mesor': rhythm_param['mesor'], 'peaks': rhythm_param['locs'], 'heights': rhythm_param['heights'],
            'llr_pvalue': p,
            'RSS': RSS, 'AIC': aic, 'BIC': bic,
            'log_likelihood': results.llf, 'logs': logs, 'mean(est)': Y_fit.mean(), 'Y(est)': Y_fit}


def get_best_n_components(df_results, model_type=None):
    if model_type:
        df_results = df_results[df_results['model_type'] == model_type].copy()

    df_results = df_results.sort_values(by='n_components')

    i = 0
    for index, new_row in df_results.iterrows():
        if i == 0:
            best_row = new_row
            i = 1
        else:
            if best_row['n_components'] == new_row['n_components']:  # non-nested
                best_row = vuong_test(best_row, new_row)
            else:  # nested
                best_row = f_test(best_row, new_row)

    return best_row


def get_best_model_type(df_results, test, n_components=None):
    if n_components:
        df_results = df_results[df_results['n_components'] == n_components].copy()

    df_results = df_results.sort_values(by='model_type')
    i = 0
    for index, new_row in df_results.iterrows():
        if i == 0:
            best_row = new_row
            i = 1
        else:
            if test == 'Vuong':
                best_row = vuong_test(best_row, new_row)
            elif test == 'F':
                best_row = f_test(best_row, new_row)
            else:
                raise Exception("Invalid criterium option.")

    return best_row


def vuong_test(first_row, second_row):
    n_points = len(first_row['logs'])
    DF1 = first_row.n_components * 2 + 1
    DF2 = second_row.n_components * 2 + 1
    DoF = DF2 - DF1

    LR = second_row['log_likelihood'] - first_row['log_likelihood'] - (DoF / 2) * math.log(n_points, 10)
    var = (1 / n_points) * sum((second_row['logs'] - first_row['logs']) ** 2)
    Z = LR / math.sqrt(n_points * var)

    v = 1 - stats.norm.cdf(Z, DoF, DF1)
    if v < 0.05:
        return second_row
    return first_row


def f_test(first_row, second_row):
    n_points = len(first_row['logs'])
    RSS1 = first_row.RSS
    RSS2 = second_row.RSS
    DF1 = n_points - (first_row.n_components * 2 + 1)
    DF2 = n_points - (second_row.n_components * 2 + 1)

    if DF2 < DF1:
        F = ((RSS1 - RSS2) / (DF1 - DF2)) / (RSS2 / DF2)
        f = 1 - stats.f.cdf(F, DF1 - DF2, DF2)
    else:
        F = ((RSS2 - RSS1) / (DF2 - DF1)) / (RSS1 / DF1)
        f = 1 - stats.f.cdf(F, DF2 - DF1, DF1)

    if f < 0.05:
        return second_row

    return first_row


def calculate_confidence_intervals_parameters(df, n_components, model_type, all_peaks, repetitions=20,
                                              precision_rate=2):
    sample_size = round(df.shape[0] - df.shape[0] / 3)
    for i in range(0, repetitions):
        sample = df.sample(sample_size)
        _, df_result, _, _, _ = fit_to_model(sample, n_components, model_type)
        if i == 0:
            amplitude = np.array(df_result['amplitude'])
            mesor = np.array(df_result['mesor'])
            peaks = np.empty((repetitions, 24))
            peaks[:] = np.nan
            peaks = hlp.add_to_table(peaks, df_result['peaks'], i)
            heights = np.empty((repetitions, 24))
            heights[:] = np.nan
            heights = hlp.add_to_table(heights, df_result['heights'], i)

        else:
            amplitude = np.append(amplitude, df_result['amplitude'])
            mesor = np.append(mesor, df_result['mesor'])
            peaks = hlp.add_to_table(peaks, df_result['peaks'], i)
            heights = hlp.add_to_table(heights, df_result['heights'], i)

    mean_amplitude = amplitude.mean()
    std_amplitude = amplitude.std()
    mean_mesor = mesor.mean()
    std_mesor = mesor.std()
    mean_std_peaks, mean_std_heights = hlp.calculate_mean_std(peaks, heights, all_peaks, precision_rate)

    amplitude = np.array([mean_amplitude - 1.96 * std_amplitude, mean_amplitude + 1.96 * std_amplitude])
    mesor = np.array([mean_mesor - 1.96 * std_mesor, mean_mesor + 1.96 * std_mesor])
    if (len(mean_std_peaks) == 0):
        peaks = []
        heights = []
    elif isinstance(mean_std_peaks[0], np.ndarray):
        peaks = np.array([mean_std_peaks[:, 0] - 1.96 * mean_std_peaks[:, 1],
                          mean_std_peaks[:, 0] + 1.96 * mean_std_peaks[:, 1]])
        heights = np.array([mean_std_heights[:, 0] - 1.96 * mean_std_heights[:, 1],
                            mean_std_heights[:, 0] + 1.96 * mean_std_heights[:, 1]])
    else:
        peaks = np.array([mean_std_peaks[0] - 1.96 * mean_std_peaks[1],
                          mean_std_peaks[0] + 1.96 * mean_std_peaks[1]])
        heights = np.array([mean_std_heights[0] - 1.96 * mean_std_heights[1],
                            mean_std_heights[0] + 1.96 * mean_std_heights[1]])

    peaks = np.transpose(peaks)
    heights = np.transpose(heights)
    return {'amplitude_CIs': np.around(amplitude, decimals=2), 'mesor_CIs': np.around(mesor, decimals=2),
            'peaks_CIs': np.around(peaks, decimals=2), 'heights_CIs': np.around(heights, decimals=2)}


def compare_by_component(df, component, n_components, models_type, ax_indices, ax_titles, rows=1, cols=1, labels=None,
                         precision_rate=2,
                         repetitions=20, save_file_to='primerjava.pdf', multiple_cols=False, main_name=''):
    df_results = pd.DataFrame()
    if multiple_cols:
        names = component
        main_name = main_name
    else:
        names = df[component].unique()
        main_name = component

    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    i = 0
    for name in names:
        if multiple_cols:
            df_name = df[['X', name]]
            df_name.columns = ['X', 'Y']
        else:
            df_name = df[df[component] == name]

        # fit
        best = find_best_model(df_name, models_type, n_components)

        model_type = best.model_type
        n_component = int(best.n_components)
        _, _, X_test, Y_test, _ = fit_to_model(df_name, n_component, model_type)

        CIs_params = calculate_confidence_intervals_parameters(df_name, n_component, model_type, best['peaks'],
                                                               repetitions=repetitions, precision_rate=precision_rate)
        # plot
        ax = fig.add_subplot(gs[ax_indices[i]])
        if name in ['ponedeljek', 'torek', 'sreda', 'cetrtek']:
            subplot_model(df_name['X'], df_name['Y'], X_test, Y_test, ax, raw_data_label='delovnik, ostali dnevi',
                          plot_model=True, color=colors[i], label=name)
        elif labels!=None:
            subplot_model(df_name['X'], df_name['Y'], X_test, Y_test, ax, plot_measurements_with_color=True,
                          plot_model=True, color=colors[i], label=labels[name])
        else:
            subplot_model(df_name['X'], df_name['Y'], X_test, Y_test, ax, plot_measurements_with_color=True,
                          plot_model=True, color=colors[i], label=name)

        best = best.to_dict()
        best[main_name] = name
        best.update(CIs_params)
        df_results = df_results.append(best, ignore_index=True)
        i = i + 1

    ax_list = fig.axes
    i = 0
    for ax in ax_list:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='large')
        ax.set_title(ax_titles[i])
        i = i + 1
    fig.tight_layout()
    plt.show()

    # save
    fig.savefig(r'results\/' + save_file_to)

    return df_results


def subplot_confidence_intervals(df, n_components, model_type, ax, repetitions=20):
    results, stats, X_test, Y_test, X_fit_test = fit_to_model(df, n_components, model_type)

    # CI
    res2 = copy.deepcopy(results)
    params = res2.params
    CIs = calculate_confidence_intervals(df, n_components, model_type, repetitions)

    N2 = round(10 * (0.7 ** n_components) + 4)
    P = np.zeros((len(params), N2))

    i = 0
    for index, CI in CIs.iterrows():
        P[i, :] = np.linspace(CI[0], CI[1], N2)
        i = i + 1

    param_samples = hlp.lazy_cartesian_product(P)
    size = param_samples.max_size
    N = round(df.shape[0] - df.shape[0] / 3)

    for i in range(0, N):
        j = random.randint(0, size)
        p = param_samples.entry_at(j)
        res2.initialize(results.model, p)
        if model_type == 'zero_nb' or model_type == "zero_poisson":
            Y_test_CI = res2.predict(X_fit_test, exog_infl=X_fit_test)
        else:
            Y_test_CI = res2.predict(X_fit_test)
        if i == 0:
            ax.plot(X_test, Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1)
        else:
            ax.plot(X_test, Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1)
