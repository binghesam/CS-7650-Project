import pandas as pd
result_file = "./test_result.csv"
df = pd.read_csv(result_file, header=None)
neutral_num = sum((df[0] < 0.5).astype(int))
print("the number of neutral sentences is: %d"%neutral_num)