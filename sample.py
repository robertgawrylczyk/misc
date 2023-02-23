# podział zbioru
random.seed(42)

train_fraction   = 0.6
validation_fraction   = 0.9 - train_fraction
test_fraction   = 0.1

random_list = random.sample(range(1, len(df) + 1), len(df))
sample_list = np.divide(random_list, len(df))

train_data         = df[(sample_list <= 1) & (sample_list > 1 - train_fraction)]
validation_data    = df[(sample_list <= 1 - train_fraction) & (sample_list > test_fraction)]
test_data          = df[(sample_list <= test_fraction) & (sample_list >= 0)]

print(len(train_data), len(validation_data), len(test_data), len(train_data) + len(validation_data) + len(test_data) == len(df))

train_data_indices = np.random.choice(df.index, len(train_data), replace = False)
train_data = train_data.append(df.loc[train_data_indices]).sort_index()

df = pd.concat([df, df.loc[train_data_indices]]).drop_duplicates(keep=False)

validation_data_indices = np.random.choice(df.index, len(validation_data), replace = False)
validation_data = validation_data.append(df.loc[validation_data_indices]).sort_index()

df = pd.concat([df, df.loc[validation_data_indices]]).drop_duplicates(keep=False)

test_data_indices = np.random.choice(df.index, len(test_data), replace = False)
test_data = test_data.append(df.loc[test_data_indices]).sort_index()

df = pd.concat([train_data, validation_data, test_data]).sort_index()

print(len(set(test_data.index)) == len(test_data.index))
print(len(set(train_data.index)) == len(train_data.index))
print(len(set(validation_data.index)) == len(validation_data.index))

print(test_data['rain_tomorrow'].sum()*2 == len(test_data))
print(validation_data['rain_tomorrow'].sum()*2 == len(validation_data))
print(test_data['rain_tomorrow'].sum()*2 == len(test_data))

print(len(train_data), len(validation_data), len(test_data), len(train_data) + len(validation_data) + len(test_data) == len(df))
print(len(df))
