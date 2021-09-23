import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

# Function to pre-process the raw data
def pre_process(data):
    # Use only 1st innings data
    data = data[data.Innings == 1]
    # Consider only those games which were completed and not interrupted by rain
    needed = pd.DataFrame({'include' : data.groupby(data['Match'])['Over'].count() == 50,
                        'include 1': data.groupby(data['Match'])['Wickets.in.Hand'].min() == 0}).reset_index()
    needed['needed'] = needed['include'] | needed['include 1']
    needed_matches = list(needed[needed['needed'] == True].Match)
    include_index = []
    for i in needed_matches:
        include_index += list(data[data.Match == i].index)
    data = data.loc[include_index, :]
    # Store cummulative runs scored in 'Total.Runs' column
    data['Total.Runs'] = data.groupby(data['Match'])['Runs'].cumsum()
    # Fetch the total scored in each match and also the number of overs bowled in each match
    total_run_look_up = pd.DataFrame({'innings total': data.groupby(data['Match'])['Total.Runs'].max(),
                                 'overs': data.groupby(data['Match'])['Over'].max()}).reset_index()
    total_runs = []
    for i in range(len(total_run_look_up)):
        total_runs += [total_run_look_up.iloc[i, 1]]*total_run_look_up.iloc[i, 2]
    data['Innings.Total.Runs'] = total_runs
    # Update 'Runs.Remaining' column
    data['Runs.Remaining'] = data['Innings.Total.Runs'] - data['Total.Runs']
    # Create 'Overs.Remaining' column
    data['Overs.Remaining'] = 50 - data['Over']
    # Create data frame consisting of only these columns - 'Match', 'Runs.Remaining', 'Wickets.in.Hand','Overs.Remaining', 'Innings.Total.Runs' 
    data = pd.DataFrame(data, columns=['Match', 'Runs.Remaining', 'Wickets.in.Hand','Overs.Remaining', 'Innings.Total.Runs'])

    return data, needed_matches

# Function for the model assumed in 1st part
def model1(Z0, b, u):
    term = Z0 * (1 - np.exp(-b*u))
    return term

# Function for the model assumed in 2nd part
def model2(Z0, L, u):
    term = Z0 * (1 - np.exp(-L*u/Z0))
    return term

# Function to compute the mean of maximum runs at given wicket
def mean_max_runs_at_given_wicket(data, wickets):
    data = data[data['Wickets.in.Hand'] == wickets]
    max_runs = data.groupby(['Match'])['Runs.Remaining'].max()
    return np.mean(max_runs)

# Part 1 function
def DuckworthLewis20Params(filepath):
    
    # Read the csv file
    data = pd.read_csv(filepath)

    # Pre-process the raw data
    data, needed_matches = pre_process(data)

    train_runs = data['Runs.Remaining'].values
    train_overs = data['Overs.Remaining'].values
    train_wickets = data['Wickets.in.Hand'].values

    w = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Initialize the values of b and Z0
    b = [0]*10
    Z0 = []
    for i in w:
        Z0.append(mean_max_runs_at_given_wicket(data, i))
    
    # Function to compute the error function per wicket
    def error(params, ww):
        bb = params[0]
        ZZ0 = params[1]
        mse = 0
        c = 0
        for i in range(len(train_runs)):
            if train_wickets[i] == ww:
                z = model1(ZZ0, bb, train_overs[i])
                mse += (z - train_runs[i])**2
                c += 1
        mse /= c
        
        # The 50th over data points are not present in the dataframe. 
        # Hence, compute the error over these points for the 10th wicket only.
        mse2 = 0
        if ww == 10:
            for i in needed_matches:
                runs = list(data[(data['Match'] == i) & (data['Overs.Remaining'] == 49)]['Innings.Total.Runs'])[0]
                z = model1(ZZ0, bb, 50)
                mse2 += (z - runs)**2
            mse2 /= len(needed_matches)
        mse += mse2
        return mse
    # Loop over all the 10 wickets
    for i in range(10):
        params = [b[i], Z0[i]]
        # Perform optimization
        opt = optimize.minimize(error, params, args=(w[i]), method='L-BFGS-B')
        b[i], Z0[i] = opt.x[0], opt.x[1]
        print('MSE for wickets in hand = ' + str(w[i]) + ' -> ' + str(opt.fun))
    return Z0, b

# Part 2 function
def DuckworthLewis11Params(filepath):

    # Read the csv file
    data = pd.read_csv(filepath)

    # Pre-process the raw data
    data, needed_matches = pre_process(data)

    train_runs = data['Runs.Remaining'].values
    train_overs = data['Overs.Remaining'].values
    train_wickets = data['Wickets.in.Hand'].values

    w = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Initialize the values of b and Z0
    parameters = []
    for i in w:
        parameters.append(mean_max_runs_at_given_wicket(data, i))
    parameters.append(15)

    # Function to compute the error function per wicket
    def error(parameters):
        Z0 = parameters[:10]
        L = parameters[10]
        mse = 0
        for i in range(len(train_runs)):
            z = model2(Z0[train_wickets[i]-1], L, train_overs[i])
            mse += (z - train_runs[i])**2
        mse /= len(train_runs)
        # The 50th over data points are not present in the dataframe. 
        # Hence, compute the error over these points for the 10th wicket only.
        mse2 = 0
        for i in needed_matches:
            runs = list(data[(data['Match'] == i) & (data['Overs.Remaining'] == 49)]['Innings.Total.Runs'])[0]
            z = model2(Z0[9], L, 50)
            mse2 += (z - runs)**2
        mse2 /= len(needed_matches)
        mse += mse2
        return mse

    # Perform optimization
    opt = optimize.minimize(error, parameters, method='L-BFGS-B')
    print('MSE -> ' + str(opt.fun))
    L_out = opt.x[-1]
    Z0_out = opt.x[:-1]
    return Z0_out, L_out


filepath = '04_cricket_1999to2011.csv'

# Part 1
print("Part 1 running...")
Z0, b = DuckworthLewis20Params(filepath)

# Plot the curves for Part 1
u = np.linspace(0, 50, num=300)
Z_final = []
for i in range(10):
    Z_final.append(model1(Z0[i], b[i], u))

plt.figure()
for i in range(10):
    plt.plot(u, Z_final[i])
plt.title("Part 1 - Average Runs obtainable")
plt.xlabel('Overs remaining')
plt.ylabel('Average Runs obtainable')
plt.xlim((0, 50))
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.grid()
plt.show()

max_val = model1(Z0[9], b[9], 50)
plt.figure()
for i in range(10):
    plt.plot(u, Z_final[i]/max_val*100)
plt.title("Part 1 - Percentage of resource remaining")
plt.xlabel('Overs remaining')
plt.ylabel('Percentage of resource remaining')
plt.xlim((0, 50))
plt.ylim((0, 1))
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.grid()
plt.show()

# Part 2
print("Part 2 running...")
Z0, L = DuckworthLewis11Params(filepath)

# Plot the curves for Part 2
u = np.linspace(0, 50, num=300)
Z_final = []
for i in range(10):
    Z_final.append(model2(Z0[i], L, u))

plt.figure()
for i in range(10):
    plt.plot(u, Z_final[i])
plt.title("Part 2 - Average Runs Obtainable")
plt.xlabel('Overs remaining')
plt.ylabel('Average Runs Obtainable')
plt.xlim((0, 50))
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.grid()
plt.show()

max_val = model2(Z0[9], L, 50)
plt.figure()
for i in range(10):
    plt.plot(u, Z_final[i]/max_val*100)
plt.title("Part 2 - Percentage of resource remaining")
plt.xlabel('Overs remaining')
plt.ylabel('Percentage of resource remaining')
plt.xlim((0, 50))
plt.ylim((0, 1))
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.grid()
plt.show()

# Slopes comparision
print("Slopes comparison at u=0")
slopes_1 = b
slopes_2 = L/Z0
mse_slopes = 1/10*np.sum((slopes_1-slopes_2)**2)
print("MSE between slopes at u=0 obtained in part1 and part2 = " + str(mse_slopes))
