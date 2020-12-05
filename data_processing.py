from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import gridspec
from sklearn.metrics import r2_score, mean_squared_error
import copy
import random
import helpers as hlp

colors = ['blue', 'green', 'orange', 'red', 'purple', 'olive', 'tomato', 'yellow']

def find_best_model(df, models_type, n_components, save_file_to='model'):
    df_results = pd.DataFrame(
        columns=['model_type', 'n_components', 'period', 'RSS', 'RMSE', 'AIC', 'BIC', 'R2',
                 'log_likelihood', 'period(est)', 'amplitude', 'acrophase', 'mesor'])

    fig = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(3, 2)

    i = 0
    for model_type in models_type:
        c = 0
        for n_component in n_components:
            results, stats, X_test, Y_test = fit_to_model(df, n_component, model_type)

            # plot
            ax = fig.add_subplot(gs[i])
            title = hlp.get_slo_model_name(model_type)
            subplot_model(df['X'], df['Y'], X_test, Y_test, ax, color=colors[c], title=title,label='št. komponent: ' + str(n_component))

            df_results = df_results.append(stats, ignore_index=True)
            c = c + 1

        i = i + 1

    # show plots
    ax_list = fig.axes
    for ax in ax_list:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize='small')

    fig.tight_layout()
    fig.savefig(r'results\/' + save_file_to + '.pdf')
    plt.show()

    # eval
    df_best = get_best_model(df_results, df.shape[0])

    return df_best


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


def fit_to_model(df, n_components, model_type,maxiter=5000,maxfun=5000,method='nm'):

    X_fit, X_test, X_fit_test, X_fit_eval_params = cosinor(df['X'], n_components=n_components, period=24)
    Y = df['Y'].to_numpy()

    X_fit = sm.add_constant(X_fit, has_constant='add')
    X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

    if model_type == 'poisson':
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method)
    elif model_type == 'gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method)
    elif model_type == 'zero_poisson':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(endog=Y, exog=X_fit, exog_infl=X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method)
    elif model_type == 'zero_nb':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(endog=Y, exog=X_fit, exog_infl=X_fit,
                                                                               p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method)
    elif model_type == 'nb':
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method)
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
    stats = calculate_statistics(Y, Y_fit, n_components, results, model_type, rhythm_params)

    return results, stats, X_test, Y_test

def calculate_CIs(df,n_components,model_type,repetitions):
    sample_size=round(df.shape[0]-df.shape[0]/3)
    for i in range(0,repetitions):
        sample=df.sample(sample_size)
        results, _, _, _=fit_to_model(sample,n_components,model_type)
        if i==0:
            save=pd.DataFrame({str(i):results.params})
        else:
            save[str(i)]=results.params

    columns = save.shape[0]

    mean = save.mean(axis=1)
    std=save.std(axis=1)
    save=pd.DataFrame({"mean":mean,"std":std})
    save['CI1']=save['mean']-1.96*save['std']
    save['CI2'] = save['mean'] + 1.96 * save['std']

    CIs = pd.DataFrame({0: [], 1: []})
    for i in range(columns):
        CIs=CIs.append({0: save['CI1'].iloc[i],1:save['CI2'].iloc[i]},ignore_index=True)

    return CIs


def confidential_intervals_of_model(df, n_components, model_type, ax, title, repetitions=30,method='nm',maxiter=5000,maxfun=5000):
    X_fit, X_test, X_fit_test, X_fit_eval_params = cosinor(df['X'], n_components=n_components, period=24)
    Y = df['Y']

    X_fit = sm.add_constant(X_fit, has_constant='add')
    X_fit_test = sm.add_constant(X_fit_test, has_constant='add')

    if model_type == 'poisson':
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method)
    elif model_type == 'gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method)
    elif model_type == 'zero_poisson':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(endog=Y, exog=X_fit, exog_infl=X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method)
    elif model_type == 'zero_nb':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(endog=Y, exog=X_fit, exog_infl=X_fit,
                                                                               p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method)
    elif model_type == 'nb':
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method)
    else:
        print("Invalid option")
        return

    if model_type == 'zero_nb' or model_type == "zero_poisson":
        Y_test = results.predict(X_fit_test, exog_infl=X_fit_test)
    else:
        Y_test = results.predict(X_fit_test)

    # CI
    res2 = copy.deepcopy(results)
    params = res2.params
    CIs = calculate_CIs(df,n_components,model_type,repetitions)

    # N = 512
    N = 4096

    if n_components == 1:
        # N2 = 8
        N2 = 12
    else:
        # N2 = 8 - n_components
        N2 = 12 - n_components

    P = np.zeros((len(params), N2))

    i = 0
    for index, CI in CIs.iterrows():
        P[i, :] = np.linspace(CI[0], CI[1], N2)
        i = i + 1

    param_samples = hlp.lazy_cartesian_product(P)
    size = param_samples.max_size
    N = min(N, size)

    for i in range(0, N):
        ix = random.randint(0, size)
        p = param_samples.entry_at(ix)
        res2.initialize(results.model, p)
        if model_type == 'zero_nb' or model_type == "zero_poisson":
            Y_test_CI = res2.predict(X_fit_test, exog_infl=X_fit_test)
        else:
            Y_test_CI = res2.predict(X_fit_test)
        if i == 0:
            ax.plot(X_test, Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1, label='intervali zaupanja modela')
        else:
            ax.plot(X_test, Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1)

    subplot_model(df['X'], Y, X_test, Y_test, ax, title=title, plot_model=False)


def subplot_model(X, Y, X_test, Y_test, ax, plot_measurements_with_color=False, plot_model=True, title='',raw_data_label='', color='black',label=''):
    ax.set_title(title)
    ax.set_xlabel('Čas [h]')
    ax.set_ylabel('Število vozil')

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


def find_alpha(df, n_components):
    # Poisson regression
    X_fit, _, X_fit_test, _ = cosinor(df['X'], n_components=n_components, period=24)
    X_fit = sm.add_constant(X_fit, has_constant='add')
    model = sm.GLM(df['Y'], X_fit, family=sm.families.Poisson())
    results = model.fit(maxiter=5000, maxfun=5000)

    # Neg. binomial
    X_fit, _, _, _ = cosinor(df['X'], n_components)

    temp_df = pd.DataFrame()
    for i in range(0, np.shape(X_fit)[1]):
        ix = str(i)
        temp_df[ix] = X_fit[:, i]

    temp_df['count'] = df['Y']
    temp_df['bb_lambda'] = results.mu
    temp_df['aux_ols_dep'] = temp_df.apply(lambda x: ((x['count'] - x['bb_lambda']) ** 2 - x['count']) / x['bb_lambda'],
                                           axis=1)
    ols_expr = """aux_ols_dep ~ bb_lambda - 1"""
    aux_olsr_results = smf.ols(ols_expr, temp_df).fit()

    # can alpha be zero? is alpha statistically significant? (is variance=mean)
    t_value = stats.t.ppf(1 - 0.99, df.shape[0])

    if t_value < aux_olsr_results.tvalues[0]:
        return aux_olsr_results.params[0]
    else:
        return 0


def evaluate_rhythm_params(X, Y):
    X = X[:240]
    Y = Y[:240]
    m = min(Y)
    M = max(Y)
    A = M - m
    MESOR = m + A / 2
    AMPLITUDE = A / 2

    PERIOD = 0
    PHASE = 0

    locs, heights = signal.find_peaks(Y, height=M * 0.75)
    heights = heights['peak_heights']

    if len(locs) >= 2:
        PERIOD = X[locs[1]] - X[locs[0]]
        PERIOD = int(round(PERIOD))

    if len(locs) >= 1:
        PHASE = X[locs[0]]

    if PERIOD:
        ACROPHASE = hlp.phase_to_radians(PHASE, PERIOD)
    else:
        ACROPHASE = np.nan

    x = np.take(X, locs)
    result = {'period': PERIOD, 'amplitude': AMPLITUDE, 'acrophase': ACROPHASE, 'mesor': MESOR, 'locs': x,
              'heights': heights}
    return result


def calculate_statistics(Y, Y_fit, n_components, results, model_type, rhythm_param):
    # p value
    # statistics according to Cornelissen (eqs (8) - (9))
    MSS = sum((Y_fit - Y.mean()) ** 2)
    RSS = sum((Y - Y_fit) ** 2)

    n_params = n_components * 2 + 1
    N = Y.size

    F = (MSS / (n_params - 1)) / (RSS / (N - n_params))
    p = 1 - stats.f.cdf(F, n_params - 1, N - n_params)

    # Another measure that describes goodnes of fit
    # How well does the curve describe the data?
    # signal to noise ratio
    # fitted curve: signal
    # noise:
    stdev_data = np.std(Y, ddof=1)
    stdev_fit = np.std(Y_fit, ddof=1)
    SNR = stdev_fit / stdev_data

    # AIC
    aic = results.aic

    # BIC
    bic = results.bic

    # RMSE
    rmse = mean_squared_error(Y, Y_fit)

    # R2
    r2 = r2_score(Y, Y_fit)

    return {'model_type': model_type, 'n_components': n_components, 'period': 24, 'p': p, 'SNR': SNR, 'RSS': RSS,'RMSE': rmse,'R2': r2, 'AIC': aic, 'BIC': bic, 'log_likelihood': results.llf, 'period(est)': rhythm_param['period'],'amplitude': rhythm_param['amplitude'],'acrophase': rhythm_param['acrophase'],'mesor': rhythm_param['mesor']}


def compare_by_criterium(fist_row, second_row):
    fist_row = fist_row.to_frame()
    second_row = second_row.to_frame()
    fist_row = fist_row.transpose()
    second_row = second_row.transpose()
    df = fist_row.append(second_row)

    df['votes'] = 0
    criteriums = ['RMSE', 'R2']

    for criterium in criteriums:
        take_min = hlp.criterium_value(criterium)
        if take_min:
            M = df[criterium].min()
            df.loc[(df[criterium] == M), 'votes'] += 1
        else:
            M = df[criterium].max()
            df.loc[(df[criterium] == M), 'votes'] += 1

    M = df['votes'].max()
    best_row = df[df['votes'] == M]
    best_row = best_row.transpose().squeeze()
    return best_row


def get_best_model(df_models, n_points):
    df_models = df_models.sort_values(by='n_components')
    i = 0
    for index, new_row in df_models.iterrows():
        if i == 0:
            best_row = new_row
            i = 1
        else:
            DF1 = best_row.n_components * 2 + 1
            DF2 = new_row.n_components * 2 + 1

            # same number of components, compare by criterium
            if DF1 == DF2:
                best_row = compare_by_criterium(best_row, new_row)

            # perform LR and F test
            else:
                p = lr_test(new_row, best_row)

                f = f_test(best_row, new_row, n_points)
                if f < 0.05 and p < 0.05:
                    best_row = new_row

    return best_row


def lr_test(first_row, second_row):
    DF1 = first_row.n_components * 2 + 1
    DF2 = second_row.n_components * 2 + 1

    LR = 2 * (first_row['log_likelihood'] - second_row['log_likelihood'])
    p = 1 - stats.chi2.cdf(LR, DF1 - DF2)

    return p


def f_test(first_row, second_row, n_points):
    RSS1 = first_row.RSS
    RSS2 = second_row.RSS
    DF1 = n_points - (first_row.n_components * 2 + 1)
    DF2 = n_points - (second_row.n_components * 2 + 1)

    if DF2 < DF1:
        F = ((RSS1 - RSS2) / (DF1 - DF2)) / (RSS2 / DF2)
        return 1 - stats.f.cdf(F, DF1 - DF2, DF2)
    else:
        F = ((RSS2 - RSS1) / (DF2 - DF1)) / (RSS1 / DF1)
        return 1 - stats.f.cdf(F, DF2 - DF1, DF1)


def compare_by_component(df, component, n_components, model_type, ax_indices, ax_titles, plot_colors, rows=1,cols=1, labels=None, save_name='primerjava.pdf', multiple_cols=False):
    if multiple_cols:
        names = component
    else:
        names = df[component].unique()
    fig = plt.figure(figsize=(8 * rows, 8 * cols))
    gs = gridspec.GridSpec(cols, rows)

    i = 0
    for name in names:
        if multiple_cols:
            df_name = df[['X', name]]
            df_name.columns = ['X', 'Y']
        else:
            df_name = df[df[component] == name]

        _, _, X_test, Y_test = fit_to_model(df_name, n_components, model_type)

        ax = fig.add_subplot(gs[ax_indices[i]])
        if labels:
            subplot_model(df_name['X'], df_name['Y'], X_test, Y_test, ax, color=colors[i],
                      plot_measurements_with_color=plot_colors[i], label=labels[name])
        else:
            subplot_model(df_name['X'], df_name['Y'], X_test, Y_test, ax, color=colors[i],
                          plot_measurements_with_color=plot_colors[i], label=name)

        i = i + 1

    ax_list = fig.axes
    i=0
    for ax in ax_list:
        ax.set_title(ax_titles[i])
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        i=i+1

    fig.tight_layout()
    fig.savefig(r'results\/' + save_name + '.pdf')
    plt.show()


