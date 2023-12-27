import pandas as pd
import matplotlib.pyplot as plt

DQN_df = pd.read_csv('output_DQN_200episodes.csv')
DDQN_df = pd.read_csv('output_DDQN_200episodes.csv')
DDQN_PERdf = pd.read_csv('output_DDQNwPER_200episodes.csv')
DuellingDDQN_df = pd.read_csv('output_DuelingDDQN_200episodes.csv')

avgDQN = DQN_df.loc[50:199]['Score'].mean()
avgDDQN = DDQN_df.loc[50:199]['Score'].mean()
avgDDQN_PER = DDQN_PERdf.loc[50:199]['Score'].mean()
avgDuellingDDQN = DuellingDDQN_df.loc[50:199]['Score'].mean()

ax = plt.gca()

DQN_df.plot(kind='scatter',
        x='Episode No',
        y='Score',
        color='red', ax=ax, label='DQN')

DDQN_df.plot(kind='scatter',
        x='Episode No',
        y='Score',
        color='green', ax=ax, label='DDQN')

DuellingDDQN_df.plot(kind='scatter',
        x='Episode No',
        y='Score',
        color='yellow', ax=ax, label='Duelling DDQN')

DDQN_PERdf.plot(kind='scatter',
        x='Episode No',
        y='Score',
        color='blue', ax=ax, label='DDQN_PER')

# set the title
plt.title('Snake Agent - Score across DQN Algorithms')
plt.legend(loc='upper left')
# show the plot
plt.show()

algos = ['DQN', 'DDQN', 'DDQN with PER', 'Duelling DQN']
values = [avgDQN, avgDDQN, avgDDQN_PER, avgDuellingDDQN]

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(algos, values, color='maroon',
        width=0.4)

plt.xlabel("Algorithms")
plt.ylabel("Avg score between episodes 51 and 200")
plt.title("Snake agent avg score")
plt.show()