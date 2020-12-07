import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import data_processing as dproc
from collections import OrderedDict
import matplotlib.dates as mdates
import numpy as np
import matplotlib.dates as md
import math

class lazy_cartesian_product:
    def __init__(self, sets):
        self.sets = sets
        self.divs = []
        self.mods = []
        self.max_size = 1
        self.precompute()

    def precompute(self):
        for i in self.sets:
            self.max_size = self.max_size * len(i)
        length = len(self.sets)
        factor = 1
        for i in range((length - 1), -1, -1):
            items = len(self.sets[i])
            self.divs.insert(0, factor)
            self.mods.insert(0, items)
            factor = factor * items

    def entry_at(self, n):
        length = len(self.sets)
        if n < 0 or n >= self.max_size:
            raise IndexError
        combination = []
        for i in range(0, length):
            combination.append(self.sets[i][int(math.floor(n / self.divs[i])) % self.mods[i]])
        return combination

def plot_models(dfs, model_type, n_components, title=[''], rows=1, cols=1, save_name='zmagovalni'):
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    i=0
    for df in dfs:
        results, stats, X_test, Y_test = dproc.fit_to_model(df, n_components[i], model_type[i])

        # plot
        ax = fig.add_subplot(gs[i])
        dproc.subplot_model(df['X'], df['Y'], X_test, Y_test, ax, color='blue', title=title[i],
                            label='prilagojena krivulja')

        i=i+1

    ax_list = fig.axes
    for ax in ax_list:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize='large')

    fig.tight_layout()
    fig.savefig(r'results\/' + save_name + '.pdf')
    plt.show()


def plot_CIs(dfs, model_type, n_components, title=[''], rows=1, cols=1, save_name='intervali',repetitions=30):
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    i = 0
    for df in dfs:
        ax = fig.add_subplot(gs[i])
        dproc.confidential_intervals_of_model(df, n_components[i], model_type[i], ax, title[i],repetitions=repetitions)

        i=i+1

    ax_list = fig.axes
    for ax in ax_list:
        ax.legend(loc='upper left',fontsize='large')

    fig.tight_layout()
    fig.savefig(r'results\/' + save_name + '.pdf')
    plt.show()


def criterium_value(criterium):
    values = dict({
        "BIC": True,
        "AIC": True,
        "RMSE": True,
        "R2": False,
        "log_likelihood": False
    })

    return values[criterium]


def plot_raw_data(file_names,title,ylabel,hour_intervals,cols=1,rows=2,save_name='izvorni'):
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    ix=0
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
        plt.xlabel('ure v dnevih [D-M H:MIN]')
        plt.ylabel(ylabel)
        plt.title(title[ix])

        ix=ix+1

    fig.tight_layout()
    fig.savefig(r'results\/' + save_name + '.pdf')
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


def get_slo_model_name(model_type):
    names = dict({
        "poisson": "Poissonov model",
        "nb": "Negativen binomski model",
        "gen_poisson": "Generaliziran Poissonov model",
        "zero_nb": "Negativen binomski model z inflacijo ničel",
        "zero_poisson": "Poissonov model z inflacijo ničel"
    })
    return names[model_type]


def phase_to_radians(phase, period=24):
    return -(phase / period) * 2 * np.pi
