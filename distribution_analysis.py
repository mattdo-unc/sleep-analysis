import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('static_R_vector_Z.csv')

X = data.iloc[:, 1:]  # Select all columns except the first one
y = data.iloc[:, 0]   # Select the first column as the target

stage_counts = y.value_counts()
print("Sleep stage counts:")
print(stage_counts)

plt.figure(figsize=(8, 6))
sns.barplot(x=stage_counts.index, y=stage_counts.values)
plt.xlabel("Sleep Stages")
plt.ylabel("Count")
plt.title("Distribution of Sleep Stages")
plt.show()

X_with_stage = X.copy()
X_with_stage['sleep_stage'] = y

summary_statistics = X_with_stage.groupby('sleep_stage').describe()
print("Summary statistics for each sleep stage:")
print(summary_statistics)

mean_connectivity = X_with_stage.groupby('sleep_stage').mean()
std_connectivity = X_with_stage.groupby('sleep_stage').std()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(mean_connectivity, cmap="coolwarm", ax=ax1)
ax1.set_title("Mean Connectivity Patterns")
ax1.set_xlabel("Neural Regions")
ax1.set_ylabel("Sleep Stages")

sns.heatmap(std_connectivity, cmap="coolwarm", ax=ax2)
ax2.set_title("Standard Deviation of Connectivity Patterns")
ax2.set_xlabel("Neural Regions")
ax2.set_ylabel("Sleep Stages")

plt.show()
