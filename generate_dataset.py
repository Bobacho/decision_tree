import random

labels = ["value1", "value2", "class"]

values: list[list[int]] = []


for i in range(1000):
    value = random.randint(0, 1000)
    value2 = random.randint(0, 1000)
    values.append([value, value2, (value + value2) % 4])


with open("data.csv", "a+") as f:
    f.write(",".join(labels) + "\n")
    for value in values:
        f.write(f"{value[0]},{value[1]},{value[2]}\n")
