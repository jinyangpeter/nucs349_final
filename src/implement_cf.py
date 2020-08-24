from src.collaborative_filtering import collaborative_filtering
from utils import data
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def load_global():
    BASE_PATH = './COVID-19/csse_covid_19_data/'
    confirmed = os.path.join(
        BASE_PATH,
        'csse_covid_19_time_series',
        'time_series_covid19_confirmed_global.csv')
    return data.load_csv_data(confirmed)


def ind_replace_by_number(confirmed_cases, number):
    threshold_ind = []
    for i in range(confirmed_cases.shape[0]):
        temp = np.argmax(confirmed_cases[i] > number)
        threshold_ind.append([True]*temp + [False]*(len(confirmed_cases[i]) - temp))
    return np.array(threshold_ind)


def ind_replace_by_percent(confirmed_cases, percent):
    threshold = confirmed_cases[:, -1] * percent
    ind = []
    for i in range(confirmed_cases.shape[0]):
        ind.append(confirmed_cases[i] <= threshold[i])
    return np.array(ind)

    # # Get the date on which a region first reported any case
    # first_case = np.argmax((confirmed_cases != 0), axis=1)
    # threshold_ind = []
    # # Get the corresponding threshold number of cases; thresholds
    # # will be determined by the quantile used
    # for i in range(len(first_case)):
    #     temp = confirmed_cases[i][first_case[i]:]
    #     temp = np.quantile(temp, quantile)
    #     # Get the first date above threshold, and then create vector of 0, 1
    #     temp = np.argmax(confirmed_cases[i] > temp)
    #     threshold_ind.append([True]*temp + [False]*(len(confirmed_cases[i]) - temp))
    # return np.array(threshold_ind)


def dhs_growth_rate(data):
    t_0 = data[:, :-1]
    t_1 = data[:, 1:]
    denominator = 0.5*(t_0 + t_1)
    numerator = t_1 - t_0
    denominator[denominator == 0] = 1
    return numerator/denominator


def rate_to_number(rate_, raw, ind):
    rate = np.copy(rate_)
    matrix = (raw * (1 - ind)).astype(float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1] - 2, 0, -1):
            if ind[i, j - 1] == 1:
                matrix[i, j - 1] = matrix[i, j] * (1 - rate[i, j - 1]/2)/(1 + rate[i, j - 1]/2)
    return matrix

    # rate = np.copy(rate_)
    # temp = raw[:, 1:]
    # num = np.zeros(temp.shape)
    # # If rate is 2, i.e. it jumps from 0 to nonzero, just use the original raw data
    # num[rate == 2] = temp[rate == 2]
    # rate[rate == 2] = -2
    # temp = raw[:, :-1]
    # num = num + temp * (1 + rate/2)/(1 - rate/2)
    # num = np.concatenate([raw[:, 0].reshape(-1, 1), num], axis=1)
    # return num


def random_drop(data, index, rows_drop, days_drop):
    ind = np.copy(index)
    data_dropped = np.copy(data)
    np.random.seed(0)
    rows = np.random.choice(data.shape[0], rows_drop, replace=False)
    # ind_drop = np.random.randint(days_drop, size=rows_drop)
    for i in range(rows_drop):
        temp = ind[rows[i]][ind[rows[i]] == 0]
        temp[0:days_drop] = 1
        temp2 = [0] * len(ind[rows[i]] == 1) + list(temp)
        ind[rows[i]][ind[rows[i]] == 0] = temp
        data_dropped[rows[i]][temp2] = 0
    return ind, data_dropped


confirmed = load_global()
confirmed_cases = confirmed.drop(confirmed.columns[0:4], axis=1)
confirmed_cases = confirmed_cases.to_numpy()
quantile_ind = ind_replace_by_percent(confirmed_cases, 0.05)
confirmed_rate = dhs_growth_rate(confirmed_cases)
number_ind = ind_replace_by_number(confirmed_cases, 0)

number_ind_dropped, confirmed_cases_dropped = random_drop(confirmed_cases, number_ind,
                                                          int(confirmed_cases.shape[0]/10), 10)
confirmed_rate_dropped = dhs_growth_rate(confirmed_cases_dropped)

np.random.seed(0)
mse = []
for dis in ['euclidean', 'cosine', 'manhattan']:
    imputed_rate = collaborative_filtering(confirmed_rate_dropped, 3, number_ind_dropped[:, :-1],
                                            distance_measure=dis, aggregator="median")
    imputed_cases = rate_to_number(imputed_rate, confirmed_cases_dropped, number_ind_dropped)
    test_ind = number_ind_dropped.astype(int) - number_ind.astype(int)
    test_ind = test_ind.astype(bool)
    m = (confirmed_cases[test_ind] - imputed_cases[test_ind])**2
    mse.append(np.mean(m))
mse = np.array(mse)
print(mse)
print("Best is", ['euclidean', 'cosine', 'manhattan'][np.argmin(mse)])

np.random.seed(0)
mse = []
for agg in ['mean', 'mode', 'median']:
    imputed_rate = collaborative_filtering(confirmed_rate_dropped, 3, number_ind_dropped[:, :-1],
                                            distance_measure="manhattan", aggregator=agg)
    imputed_cases = rate_to_number(imputed_rate, confirmed_cases_dropped, number_ind_dropped)
    test_ind = number_ind_dropped.astype(int) - number_ind.astype(int)
    test_ind = test_ind.astype(bool)
    m = (confirmed_cases[test_ind] - imputed_cases[test_ind]) ** 2
    mse.append(np.mean(m))
mse = np.array(mse)
print(mse)
print("Best is", ['mean', 'mode', 'median'][np.argmin(mse)])

np.random.seed(0)
mse = []
for k in range(1, 31):
    imputed_rate = collaborative_filtering(confirmed_rate_dropped, k, number_ind_dropped[:, :-1],
                                            distance_measure="manhattan", aggregator="mean")
    imputed_cases = rate_to_number(imputed_rate, confirmed_cases_dropped, number_ind_dropped)
    test_ind = number_ind_dropped.astype(int) - number_ind.astype(int)
    test_ind = test_ind.astype(bool)
    m = (confirmed_cases[test_ind] - imputed_cases[test_ind]) ** 2
    mse.append(np.mean(m))
mse = np.array(mse)
print("Best is", range(1, 31)[np.argmin(mse)])

plt.title("MSE for K in range(1, 31)")
plt.ylabel("MSE")
plt.plot(range(1, 31), mse)
plt.savefig("final_mse.png")

###############################
number_ind_0 = ind_replace_by_number(confirmed_cases, 10)
imputed_rate = collaborative_filtering(confirmed_rate, 2, number_ind_0[:, :-1],
                                            distance_measure="manhattan", aggregator="mean")
imputed_cases = rate_to_number(imputed_rate, confirmed_cases, number_ind_0)
imputed_cases = np.rint(imputed_cases).astype(int)

temp = (imputed_rate != confirmed_rate)
print(np.sum(temp))
temp = (imputed_cases != confirmed_cases)
print(np.sum(imputed_cases - confirmed_cases))
print(np.sum(temp))
temp = np.argwhere(imputed_cases != confirmed_cases)

temp = imputed_cases- confirmed_cases
temp = np.sum(temp, axis=1)
print(np.sum(temp != 0))

top_name = list(confirmed[confirmed.columns[1]][np.argsort(-temp)[0:4]])
top = np.argsort(-temp)[0:4]

date = list(confirmed.columns[4:])
# date = pd.to_datetime(date, format="%m/%d/%y")
date = [d[:-3] for d in date]

cut = np.argmin(number_ind_0, axis=1)[top]

days_more = 10

for i, l in enumerate(top_name):
    fig, ax = plt.subplots()
    ax.set_title("Confirmed vs Imputed (Until 2%): " + l, fontsize=12)
    ax.set_ylabel("Total Reported Cases")
    ax.set_xlabel("Date")
    ax.plot(date[:cut[i]+days_more], confirmed_cases[top[i]][:cut[i]+days_more], color="black", label="Confirmed")
    ax.plot(date[:cut[i]+days_more], imputed_cases[top[i]][:cut[i]+days_more], "--", color="red", label="Imputed")
    tick = (len(date[:cut[i]+days_more]) // 20)
    if tick == 0:
        tick = range(0, len(date[:cut[i] + days_more]))
    else:
        tick = range(0, len(date[:cut[i]+days_more]), tick)
    plt.xticks(list(tick), [date[:cut[i]+days_more][k] for k in tick], rotation=45)
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    fig.savefig(f"{l}_2%.png", bbox_inches='tight')

total_cases = np.sum(confirmed_cases, axis=0)
total_imputed_cases = np.sum(imputed_cases, axis=0)


fig, ax = plt.subplots()
ax.plot(date, total_cases, color="black", label="Confirmed")
ax.plot(date, total_imputed_cases, "--", color="red", label="Imputed")
ax.set_title("Confirmed vs Imputed: Global Total", fontsize=16)
ax.set_ylabel('Total Reported Cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")
ax.set_yscale('log')
plt.xticks(range(date), range(date), rotation=45)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('results/cases_by_country.png')
