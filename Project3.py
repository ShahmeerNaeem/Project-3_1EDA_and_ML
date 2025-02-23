# Soccer League DataBase EDA & Insights

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import xml.etree.ElementTree as ET
from scipy import stats
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC

# Connect to the SQLite database
database_path = 'database.sqlite'
conn = sqlite3.connect(database_path)

# List all tables in the database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in the database:")
print(tables)

# Load the tables into pandas DataFrame
sqlite_sequence = pd.read_sql_query("SELECT * FROM sqlite_sequence", conn)
print('The columns for sqlite_sequence are :', sqlite_sequence.info())
print("sqlite_sequence DataFrame:")
sqlite_sequence.head()

player_attributes = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
player_attributes.drop("id", axis=1, inplace=True)
print('The columns for player_attributes are :', player_attributes.info())
print("\nPlayer Attributes DataFrame:")
player_attributes.head()

player = pd.read_sql_query("SELECT * FROM Player", conn)
player.drop("id", axis=1, inplace=True)
print('The columns for player_table are :', player.info())
print("Player DataFrame:")
player.head()

league = pd.read_sql_query("SELECT * FROM League", conn)
league.drop("id", axis=1, inplace=True)
print('The columns for league_table are :', league.info())
print("\nLeague DataFrame:")
league.head()

country = pd.read_sql_query("SELECT * FROM Country", conn)
print('The columns for country_table are :', country.info())
print("\nCountry DataFrame:")
country.head()

team = pd.read_sql_query("SELECT * FROM Team", conn)
team.drop("id", axis=1, inplace=True)
print('The columns for team_table are :', team.info())
print("\nTeam DataFrame:")
team.head()

team_attributes = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
team_attributes.drop("id", axis=1, inplace=True)
print('The columns for team_attributes are :', team_attributes.info())
team_attributes.fillna(0, inplace=True)
print("Team Attributes DataFrame:")
team_attributes.head()

match = pd.read_sql_query("SELECT * FROM Match", conn)
match.drop("id", axis=1, inplace=True)
print('The columns for match_table are :', match.info())
print("\nMatch DataFrame:")
match.head()

# ---
# Function to extract values from XML data
def extract_xml_values(xml_data, tag):
    try:
        root = ET.fromstring(xml_data)
        values = [int(value.find(tag).text) for value in root.findall('value') if value.find(tag) is not None]
        return sum(values) if values else np.nan
    except (ET.ParseError, AttributeError, TypeError):
        return np.nan

# List of XML columns and the tags to extract
xml_columns = {
    'goal': 'goal',
    'shoton': 'elapsed',
    'shotoff': 'elapsed',
    'foulcommit': 'elapsed',
    'card': 'card',
    'cross': 'cross',
    'corner': 'corner',
    'possession': 'homepos'
}

# Process each XML column
for col, tag in xml_columns.items():
    match[col] = match[col].apply(lambda x: extract_xml_values(x, tag) if pd.notnull(x) else np.nan)

# Fill NaN values with 0 for numerical consistency
match.fillna(0, inplace=True)

relevant_columns = ['country_id', 'league_id', 'season', 'stage', 'date', 'match_api_id',
                    'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal',
                    'goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession']

processed_df = match[relevant_columns]
print("\nProcessed Match DataFrame:")
processed_df.head()

# ---
# Select relevant columns from team and team_attributes
team_df_selected = team[['team_api_id', 'team_long_name']].rename(columns={'team_long_name': 'team_name'})
team_attributes_df_selected = team_attributes[['team_api_id', 'date', 'buildUpPlaySpeed', 'buildUpPlayPassing',
                                               'chanceCreationPassing', 'chanceCreationCrossing',
                                               'chanceCreationShooting', 'defencePressure', 'defenceAggression',
                                               'defenceTeamWidth']]

team_attributes_df_selected.loc[:, 'date'] = pd.to_datetime(team_attributes_df_selected['date'])
team_data = team_df_selected.merge(team_attributes_df_selected, on='team_api_id', how='left')
print(team_data.head())

# ---
# Selecting important columns from player_attributes
player_data = player_attributes[['player_api_id', 'overall_rating', 'potential', 'preferred_foot',
                                 'attacking_work_rate', 'defensive_work_rate', 'crossing', 'finishing',
                                 'dribbling', 'ball_control', 'acceleration', 'sprint_speed',
                                 'stamina', 'positioning', 'marking',
                                 'standing_tackle', 'sliding_tackle']]

# Selecting height and weight from player
player_info = player[['player_api_id', 'height', 'weight']]
player_data = player_data.merge(player_info, on='player_api_id', how='left')
print(player_data.info())
print(player_data.head())

# ---
# Function to optimize data types
def optimize_data_types(df):
    df = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

processed_df = optimize_data_types(processed_df)
country = optimize_data_types(country)
league = optimize_data_types(league)
team = optimize_data_types(team)
player = optimize_data_types(player)
player_data = optimize_data_types(player_data)
team_data = optimize_data_types(team_data)
print(processed_df.info())

# ---
# Function to display null values and rows to be dropped
def display_null_info(df, df_name):
    print(f"Null values in {df_name}:")
    print(df.isnull().sum())
    print(f"\nTotal rows in {df_name}: {len(df)}")
    print(f"Rows with any null values in {df_name}: {df.isnull().any(axis=1).sum()}\n")

display_null_info(processed_df, 'processed_df')
display_null_info(country, 'country')
display_null_info(league, 'league')
display_null_info(player_data, 'player_data')
display_null_info(team, 'team')

processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
country.replace([np.inf, -np.inf], np.nan, inplace=True)
team.replace([np.inf, -np.inf], np.nan, inplace=True)

team_df_selected = team[['team_api_id', 'team_long_name']].rename(columns={'team_long_name': 'team_name'})
country_df_selected = country[['id', 'name']].rename(columns={'id': 'country_id', 'name': 'country_name'})
team_attributes_df_selected = team_attributes[['team_api_id', 'date']].rename(columns={'date': 'team_attribute_date'})
league_df_selected = league[['country_id', 'name']].rename(columns={'name': 'league_name'})

team_attributes_df_selected['team_attribute_date'] = pd.to_datetime(team_attributes_df_selected['team_attribute_date'])
processed_df['date'] = pd.to_datetime(processed_df['date'])

merged_df1 = processed_df.merge(country_df_selected, on='country_id', how='left')
merged_df2 = merged_df1.merge(team_df_selected, left_on='home_team_api_id', right_on='team_api_id', how='left') \
    .rename(columns={'team_name': 'home_team_name'}) \
    .drop(columns=['team_api_id'])
merged_df3 = merged_df2.merge(team_df_selected, left_on='away_team_api_id', right_on='team_api_id', how='left') \
    .rename(columns={'team_name': 'away_team_name'}) \
    .drop(columns=['team_api_id'])
merged_df4 = merged_df3.merge(team_attributes_df_selected,
                              left_on=['home_team_api_id', 'date'],
                              right_on=['team_api_id', 'team_attribute_date'],
                              how='left') \
    .rename(columns={'team_attribute_date': 'home_team_attribute_date'}) \
    .drop(columns=['team_api_id'])
merged_df5 = merged_df4.merge(team_attributes_df_selected,
                              left_on=['away_team_api_id', 'date'],
                              right_on=['team_api_id', 'team_attribute_date'],
                              how='left') \
    .rename(columns={'team_attribute_date': 'away_team_attribute_date'}) \
    .drop(columns=['team_api_id'])
df = merged_df5.merge(league_df_selected, left_on='country_id', right_on='country_id', how='left') \
    .rename(columns={'league_name': 'tournament_type'}) \
    .drop(columns=['league_id'])
print(df.info())

# ---
# EDA: Question 1: Key Factors Influencing Match Outcomes
df['goal_difference'] = df['home_team_goal'] - df['away_team_goal']
df['match_result'] = df.apply(lambda row: 'Home Win' if row['home_team_goal'] > row['away_team_goal'] else
('Away Win' if row['home_team_goal'] < row['away_team_goal'] else 'Draw'), axis=1)
print("\nMatch DataFrame with Goal Difference and Match Result:")
print(df[['home_team_goal', 'away_team_goal', 'goal_difference', 'match_result']].head())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='match_result', hue='match_result', palette='viridis', legend=True)
plt.title('Distribution of Match Results')
plt.xlabel('Match Result')
plt.ylabel('Count')
plt.savefig('Distribution of Match Results.png', bbox_inches='tight')
plt.show()

# ---
# Step 2: Check Correlations between Features and Match Result
features = ['possession', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'home_team_goal', 'away_team_goal']
df_selected = df[features + ['match_result']].copy()
df_selected.loc[:, 'match_result'] = df_selected['match_result'].map({'win': 1, 'draw': 0, 'loss': -1})
correlation_matrix = df_selected.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", vmin=-1, vmax=1)
plt.title("Correlation between Features and Match Result")
plt.savefig('Correlation Heatmap.png', bbox_inches='tight')
plt.show()

# ---
# Step 1: Define the features for which you want to create boxplots
features = ['possession', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'goal_difference']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.boxplot(x='match_result', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f'{feature} vs Match Result')
plt.tight_layout()
plt.savefig('Boxplots Match Result.png', bbox_inches='tight')
plt.show()

# ---
# Extract month and year from the date column
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['total_goals'] = df['home_team_goal'] + df['away_team_goal']
monthly_goals = df.groupby('month')['total_goals'].sum()
df['home_team_win'] = df['home_team_goal'] > df['away_team_goal']
monthly_home_wins = df.groupby('month')['home_team_win'].mean()
print("Total Goals by Month:")
print(monthly_goals)
print("\nHome Team Win Rate by Month:")
print(monthly_home_wins)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
monthly_goals.plot(kind='bar', color='skyblue', title="Total Goals by Month")
plt.ylabel('Total Goals')
plt.subplot(1, 2, 2)
monthly_home_wins.plot(kind='bar', color='salmon', title="Home Team Win Rate by Month")
plt.ylabel('Home Team Win Rate')
plt.tight_layout()
plt.savefig('Monthly Goals and Home Wins.png', bbox_inches='tight')
plt.show()

# ---
# Calculate total goals and home-team win rate by season
seasonal_goals = df.groupby('season')['total_goals'].sum()
seasonal_home_wins = df.groupby('season')['home_team_win'].mean()
print("\nTotal Goals by Season:")
print(seasonal_goals)
print("\nHome Team Win Rate by Season:")
print(seasonal_home_wins)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
seasonal_goals.plot(kind='bar', color='lightgreen', title="Total Goals by Season")
plt.ylabel('Total Goals')
plt.subplot(1, 2, 2)
seasonal_home_wins.plot(kind='bar', color='orange', title="Home Team Win Rate by Season")
plt.ylabel('Home Team Win Rate')
plt.tight_layout()
plt.savefig('Home Team Win Rate.png', bbox_inches='tight')
plt.show()

# ---
# Group data by tournament type and calculate total goals and home team win rate
tournament_goals = df.groupby('tournament_type')['total_goals'].sum()
tournament_home_wins = df.groupby('tournament_type')['home_team_win'].mean()
print("Total Goals by Tournament Type:")
print(tournament_goals)
print("\nHome Team Win Rate by Tournament Type:")
print(tournament_home_wins)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
tournament_goals.plot(kind='bar', color='lightblue', title="Total Goals by Tournament Type")
plt.ylabel('Total Goals')
plt.subplot(1, 2, 2)
tournament_home_wins.plot(kind='bar', color='pink', title="Home Team Win Rate by Tournament Type")
plt.ylabel('Home Team Win Rate')
plt.tight_layout()
plt.savefig('Home Team Win Rate by Tournament Type.png', bbox_inches='tight')
plt.show()

# ---
# Aggregate home team stats (goals, matches, wins)
home_team_stats = df.groupby('home_team_api_id').agg(
    total_goals=('home_team_goal', 'sum'),
    total_matches=('home_team_api_id', 'count'),
    total_wins=('home_team_goal', lambda x: (x > df.loc[x.index, 'away_team_goal']).sum())
).reset_index().rename(columns={'home_team_api_id': 'team_api_id'})

# Aggregate away team stats (goals, matches, wins)
away_team_stats = df.groupby('away_team_api_id').agg(
    total_goals=('away_team_goal', 'sum'),
    total_matches=('away_team_api_id', 'count'),
    total_wins=('away_team_goal', lambda x: (x > df.loc[x.index, 'home_team_goal']).sum())
).reset_index().rename(columns={'away_team_api_id': 'team_api_id'})

# Combine the home and away stats into one dataframe
team_stats = pd.concat([home_team_stats, away_team_stats], ignore_index=True)
team_stats = team_stats.groupby('team_api_id').agg(
    total_goals=('total_goals', 'sum'),
    total_matches=('total_matches', 'sum'),
    total_wins=('total_wins', 'sum')
).reset_index()
team_stats['win_rate'] = team_stats['total_wins'] / team_stats['total_matches']
print(f"Number of unique teams: {team_stats['team_api_id'].nunique()}")
print(team_stats.head())

# ---
# Aggregate defensive metrics (e.g., average values per team)
defensive_stats = team_data.groupby('team_api_id').agg(
    avg_defencePressure=('defencePressure', 'mean'),
    avg_defenceAggression=('defenceAggression', 'mean'),
    avg_defenceTeamWidth=('defenceTeamWidth', 'mean')
).reset_index()
team_stats_final = pd.merge(team_stats, defensive_stats, on='team_api_id', how='left')

# Merge aggregated stats into team_data
team_data_final = pd.merge(team_data, team_stats_final[['team_api_id', 'total_goals', 'total_matches', 'total_wins', 'win_rate']],
                           on='team_api_id', how='left')

# ---
# Top 10 teams by total goals scored
top_teams_by_goals_scored = team_data_final[['team_name', 'total_goals']].groupby('team_name').sum().sort_values('total_goals', ascending=False).head(10)
print("Top 10 Teams by Total Goals Scored:")
print(top_teams_by_goals_scored)

# Top 10 teams by win rate
top_teams_by_win_rate = team_data_final[['team_name', 'win_rate']].groupby('team_name').mean().sort_values('win_rate', ascending=False).head(10)
print("\nTop 10 Teams by Win Rate:")
print(top_teams_by_win_rate)

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.barplot(x=top_teams_by_goals_scored.index, y=top_teams_by_goals_scored['total_goals'], ax=ax[0], palette='viridis')
ax[0].set_title('Top 10 Teams by Total Goals Scored')
ax[0].set_xlabel('Team Name')
ax[0].set_ylabel('Total Goals Scored')
ax[0].tick_params(axis='x', rotation=90)
sns.barplot(x=top_teams_by_win_rate.index, y=top_teams_by_win_rate['win_rate'], ax=ax[1], palette='magma')
ax[1].set_title('Top 10 Teams by Win Rate')
ax[1].set_xlabel('Team Name')
ax[1].set_ylabel('Win Rate')
ax[1].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig('Top 10 Teams by Goals and Win.png', bbox_inches='tight')
plt.show()

# ---
# Top 10 teams by average defensive pressure
top_teams_by_defencePressure = team_data_final[['team_name', 'defencePressure']].groupby('team_name').mean().sort_values('defencePressure', ascending=False).head(10)
print(top_teams_by_defencePressure)

# Top 10 teams by average defensive aggression
top_teams_by_defenceAggression = team_data_final[['team_name', 'defenceAggression']].groupby('team_name').mean().sort_values('defenceAggression', ascending=False).head(10)
print(top_teams_by_defenceAggression)

# Top 10 teams by average defensive team width
top_teams_by_defenceTeamWidth = team_data_final[['team_name', 'defenceTeamWidth']].groupby('team_name').mean().sort_values('defenceTeamWidth', ascending=False).head(10)
print(top_teams_by_defenceTeamWidth)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.barplot(x=top_teams_by_defencePressure.index, y=top_teams_by_defencePressure['defencePressure'], ax=ax[0], palette='coolwarm')
ax[0].set_title('Top 10 Teams by Defence Pressure')
ax[0].set_xlabel('Team Name')
ax[0].set_ylabel('Defence Pressure')
ax[0].tick_params(axis='x', rotation=90)
sns.barplot(x=top_teams_by_defenceAggression.index, y=top_teams_by_defenceAggression['defenceAggression'], ax=ax[1], palette='Blues')
ax[1].set_title('Top 10 Teams by Defence Aggression')
ax[1].set_xlabel('Team Name')
ax[1].set_ylabel('Defence Aggression')
ax[1].tick_params(axis='x', rotation=90)
sns.barplot(x=top_teams_by_defenceTeamWidth.index, y=top_teams_by_defenceTeamWidth['defenceTeamWidth'], ax=ax[2], palette='YlGnBu')
ax[2].set_title('Top 10 Teams by Defence Team Width')
ax[2].set_xlabel('Team Name')
ax[2].set_ylabel('Defence Team Width')
ax[2].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig('Top 10 Teams by Defensive.png', bbox_inches='tight')
plt.show()

# ---
# Select relevant columns for the correlation matrix
correlation_columns = [
    'buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing', 'chanceCreationCrossing',
    'chanceCreationShooting', 'defencePressure', 'defenceAggression', 'defenceTeamWidth',
    'total_goals', 'win_rate'
]

# Calculate the correlation matrix for these selected features
correlation_matrix = team_data_final[correlation_columns].corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix for Team Performance Metrics (Attack & Defense)')
plt.savefig('Correlation Matrix for Team Performance Metrics (Attack & Defense).png', bbox_inches='tight')
plt.show()

# ---
player_data['attacking_work_rate'] = player_data['attacking_work_rate'].fillna(player_data['attacking_work_rate'].mode()[0])
player_data['defensive_work_rate'] = player_data['defensive_work_rate'].fillna(player_data['defensive_work_rate'].mode()[0])
player_data['sliding_tackle'] = player_data['sliding_tackle'].fillna(player_data['sliding_tackle'].mean())

# Encoding categorical columns
player_data_encoded = player_data.copy()
player_data_encoded['preferred_foot'] = player_data_encoded['preferred_foot'].map({'Right': 1, 'Left': 0})
player_data_encoded['attacking_work_rate'] = player_data_encoded['attacking_work_rate'].map({'High': 1, 'Medium': 0.5, 'Low': 0})
player_data_encoded['defensive_work_rate'] = player_data_encoded['defensive_work_rate'].map({'High': 1, 'Medium': 0.5, 'Low': 0})

# Correlation Analysis
correlation_matrix = player_data_encoded[['overall_rating', 'potential', 'crossing', 'finishing', 'dribbling',
                                           'ball_control', 'acceleration', 'sprint_speed', 'stamina', 'marking',
                                           'standing_tackle', 'sliding_tackle', 'height', 'weight']].corr()

# Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title("Player Attribute Correlation Matrix")
plt.show()

# ---
# 1. Basic Statistics Calculation
match_results_stats = df['total_goals'].describe()

# Z-score method
z_scores = stats.zscore(df['total_goals'])
outliers_zscore = df[abs(z_scores) > 3]

# IQR method
Q1 = df['total_goals'].quantile(0.25)
Q3 = df['total_goals'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['total_goals'] < (Q1 - 1.5 * IQR)) | (df['total_goals'] > (Q3 + 1.5 * IQR))]

# Combine the two outliers
combined_outliers = pd.concat([outliers_zscore, outliers_iqr]).drop_duplicates().reset_index()

print(f"Total Anomalies detected (Z-score + IQR): {combined_outliers.shape[0]}")
combined_outliers[['match_api_id', 'home_team_goal', 'away_team_goal', 'total_goals', 'match_result', 'goal_difference', 'home_team_win', 'tournament_type']].head(10)

# 2. Inspecting Goal Difference and Match Outcomes
anomaly_goals = combined_outliers[['home_team_goal', 'away_team_goal', 'total_goals', 'goal_difference', 'match_result']]
anomaly_goals['goal_difference'] = anomaly_goals['home_team_goal'] - anomaly_goals['away_team_goal']
anomaly_goals = anomaly_goals[anomaly_goals['goal_difference'].abs() > 3]
print(f"Number of anomalies based on goal difference > 3: {anomaly_goals.shape[0]}")
anomaly_goals.head()

# 3. Investigating Tournament Types
tournament_anomalies = combined_outliers['tournament_type'].value_counts()
print(tournament_anomalies)

plt.figure(figsize=(12, 6))
sns.barplot(x=tournament_anomalies.index, y=tournament_anomalies.values, palette='viridis')
plt.title('Number of Anomalies per Tournament Type', fontsize=16)
plt.xlabel('Tournament Type', fontsize=14)
plt.ylabel('Number of Anomalies', fontsize=14)
plt.xticks(rotation=90, ha='right')
plt.savefig('Tournament Anomalies.png', bbox_inches='tight')
plt.show()

# 4. Investigating Match Stages
anomaly_stage = combined_outliers[['stage', 'home_team_goal', 'away_team_goal', 'total_goals']]
plt.figure(figsize=(10, 6))
sns.countplot(data=anomaly_stage, x='stage', palette='viridis')
plt.title('Anomalies by Match Stage')
plt.xticks(rotation=45)
plt.savefig('Anomalies by Match Stage.png', bbox_inches='tight')
plt.show()

# 5. Identifying Unexpected Wins/Losses
unexpected_wins = combined_outliers[combined_outliers['goal_difference'] > 3]
print(f"Number of unexpected wins/losses with goal difference > 3: {unexpected_wins.shape[0]}")
unexpected_wins[['home_team_name', 'away_team_name', 'goal_difference', 'match_result']].head()

# ---
plt.figure(figsize=(10, 6))
sns.histplot(df['total_goals'], bins=50, kde=True, color=sns.color_palette('viridis')[0])
plt.axvline(x=df['total_goals'].mean(), color='red', linestyle='--', label="Mean Total Goals")
plt.axvline(x=df['total_goals'].median(), color='green', linestyle='--', label="Median Total Goals")
plt.title('Distribution of Total Goals')
plt.legend()
plt.savefig('Distribution of Total.png', bbox_inches='tight')
plt.show()

# Now Filtering Specific Matches with Extreme Scores:
extreme_scores = combined_outliers[(combined_outliers['home_team_goal'] > 5) | (combined_outliers['away_team_goal'] > 5)]
print(extreme_scores[['home_team_name', 'away_team_name', 'home_team_goal', 'away_team_goal']].head())

plt.figure(figsize=(10, 6))
sns.histplot(combined_outliers['goal_difference'], bins=50, kde=True, color=sns.color_palette('viridis')[0])
plt.title('Goal Difference Distribution for Anomalous Matches')
plt.savefig('Goal Difference Distribution for Anomalous Matches.png', bbox_inches='tight')
plt.show()

# ---
home_win_rate = df[df['home_team_win'] == True].shape[0] / df.shape[0]
away_win_rate = df[df['home_team_win'] == False].shape[0] / df.shape[0]
home_avg_goals = df['home_team_goal'].mean()
away_avg_goals = df['away_team_goal'].mean()
home_avg_possession = df['possession'][df['home_team_win'] == True].mean()
away_avg_possession = df['possession'][df['home_team_win'] == False].mean()

home_goals = df['home_team_goal']
away_goals = df['away_team_goal']
home_possession = df['possession'][df['home_team_win'] == True]
away_possession = df['possession'][df['home_team_win'] == False]

goals_ttest = ttest_ind(home_goals, away_goals)
possession_ttest = ttest_ind(home_possession, away_possession)

print(f"Home Win Rate: {home_win_rate:.4f}")
print(f"Away Win Rate: {away_win_rate:.4f}")
print(f"Home Average Goals: {home_avg_goals:.2f}")
print(f"Away Average Goals: {away_avg_goals:.2f}")
print(f"Home Average Possession: {home_avg_possession:.2f}")
print(f"Away Average Possession: {away_avg_possession:.2f}")
print(f"T-test results for Goals: p-value = {goals_ttest.pvalue:.4f}")
print(f"T-test results for Possession: p-value = {possession_ttest.pvalue:.4f}")

# ---
metrics = ['Home Team', 'Away Team']
home_goals = [1.54]
away_goals = [1.16]
home_possession = [72.87]
away_possession = [66.45]
home_win_rate = [0.4587]
away_win_rate = [0.5413]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.barplot(x=metrics, y=[home_goals[0], away_goals[0]], ax=axes[0], palette='coolwarm')
axes[0].set_title('Home vs Away - Average Goals', fontsize=14)
axes[0].set_ylabel('Goals', fontsize=12)
sns.barplot(x=metrics, y=[home_possession[0], away_possession[0]], ax=axes[1], palette='coolwarm')
axes[1].set_title('Home vs Away - Average Possession', fontsize=14)
axes[1].set_ylabel('Possession (%)', fontsize=12)
sns.barplot(x=metrics, y=[home_win_rate[0], away_win_rate[0]], ax=axes[2], palette='coolwarm')
axes[2].set_title('Home vs Away - Win Rate', fontsize=14)
axes[2].set_ylabel('Win Rate', fontsize=12)
for ax in axes:
    ax.set_xlabel('Team', fontsize=12)
plt.suptitle('Home vs Away Team Performance Comparison', fontsize=16)
plt.tight_layout()
plt.show()

# ---
team_data_final = optimize_data_types(team_data_final)
df = optimize_data_types(df)
team_data_final.info()

# ---
# Important features from match dataset
selected_match_features = ['home_team_api_id', 'away_team_api_id',
    'possession', 'shoton', 'shotoff', 'foulcommit', 'tournament_type', 'month', 'year', 'match_result']

# Important team stats
selected_team_features = [
    'team_api_id', 'win_rate', 'total_goals', 'total_wins',
    'chanceCreationPassing' , 'chanceCreationCrossing','chanceCreationShooting' ,
    'defencePressure','defenceAggression' ,'defenceTeamWidth'
]

# Filter only required columns from team_data_final
team_data_filtered = team_data_final[selected_team_features]
df_filtered = df[selected_match_features]

# Merge home team stats
df_merged1 = df_filtered.merge(team_data_filtered, left_on='home_team_api_id', right_on='team_api_id', suffixes=('', '_home'))

# Manually rename home columns (since Pandas didn't add suffixes) Making them better
home_columns = {
    'win_rate': 'win_rate_home',
    'total_goals': 'total_goals_home',
    'total_wins': 'total_wins_home',
    'chanceCreationPassing': 'chanceCreationPassing_home',
    'chanceCreationCrossing': 'chanceCreationCrossing_home',
    'chanceCreationShooting': 'chanceCreationShooting_home',
    'defencePressure': 'defencePressure_home',
    'defenceAggression': 'defenceAggression_home',
    'defenceTeamWidth': 'defenceTeamWidth_home'
}
df_merged1 = df_merged1.rename(columns=home_columns)

# Merge team stats
df_merged2 = df_merged1.merge(team_data_filtered, left_on='away_team_api_id', right_on='team_api_id', suffixes=('', '_away'))

# Manually rename away columns (if needed)
away_columns = {
    'win_rate': 'win_rate_away',
    'total_goals': 'total_goals_away',
    'total_wins': 'total_wins_away',
    'chanceCreationPassing': 'chanceCreationPassing_away',
    'chanceCreationCrossing': 'chanceCreationCrossing_away',
    'chanceCreationShooting': 'chanceCreationShooting_away',
    'defencePressure': 'defencePressure_away',
    'defenceAggression': 'defenceAggression_away',
    'defenceTeamWidth': 'defenceTeamWidth_away'
}
df_merged2 = df_merged2.rename(columns=away_columns)

# Drop redundant columns
training_df = df_merged2.drop(columns=['team_api_id', 'team_api_id_away'])
print(training_df.head())
print(training_df.info())

# ---
target_column = 'match_result'
X = training_df.drop(columns=[target_column])
y = training_df[target_column]

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("Encoded target values (y):")
print(y[958:1049])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print("Numeric Columns:", numeric_features)
print("Categorical Columns:", categorical_features)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

numeric_transformer = Pipeline(steps=[
    ('imputer', numeric_imputer),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', categorical_imputer),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

categorical_column_names = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
new_column_names = numeric_features + list(categorical_column_names)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=new_column_names)
print(X_train_processed_df.head())

import numpy as np
print("Missing values after preprocessing in training data:")
print(np.isnan(X_train_processed).sum())

print(f"X_train shape: {X_train_processed.shape}")
print(f"X_test shape: {X_test_processed.shape}")
print(f"y_test shape: {y_test.shape}")
print(len(X_train_processed))
print(len(y))

# ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100),
    'SVM': SVC(kernel='linear', probability=True, random_state=42) 
}


for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_processed, y_train)
    with open(f'{model_name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print(f"Saved {model_name} model.")
print("All models trained and saved.")

# ---
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

trained_models = {}
for model_name in models.keys():
    with open(f'{model_name}_model.pkl', 'rb') as file:
        trained_models[model_name] = pickle.load(file)

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
axes = axes.flatten()

for i, (model_name, model) in enumerate(trained_models.items()):
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test_processed)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(f'{model_name} Confusion Matrix')
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")
    print(f"Classification Report for {model_name}:\n{report}")
    print("-" * 50)
plt.tight_layout()
plt.show()

# ---
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append([name, accuracy, precision, recall, f1])

metrics_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
print(metrics_df)