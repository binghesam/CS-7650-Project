# split the full_size to train, val, and test
# totoal full 181496 lines

full_file = "./biased.full"
tot_lines = 181496
train_file = "./biased.full.train" # full_file + ".train"
val_file = "./biased.full.val" # full_file + ".val"
test_file = "./biased.full.test" # # full_file + ".test"
with open(full_file, encoding="utf-8") as bigfile, \
        open(train_file, "w", encoding="utf-8") as train_data, \
        open(val_file, "w", encoding="utf-8") as val_data, \
        open(test_file, "w", encoding="utf-8") as test_data:
    for lineno, line in enumerate(bigfile):
        if lineno <= 10:
            train_data.write(line)
        elif lineno <= 20:
            val_data.write(line)
        elif lineno <= 30:
            test_data.write(line)
        else:
            print(lineno)
            break
