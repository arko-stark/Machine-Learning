import load_data_project as lp
X, y = lp.load_data()
print(f'Training data matrix shape: {X.shape}')
print(f'Labels vector shape: {y.shape}')