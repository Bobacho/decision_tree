import random


with open("heart.csv", "a") as f:
    for _ in range(1_000_000):
        age = random.randint(28, 77)
        gender = 1 if random.random() > 0.5 else 0
        resting = random.randint(90, 200)
        cholesterol = random.randint(175, 603)
        fasting_rm = 0 if random.random() > 0.1 else 0
        r = random.random()
        resting_egc = 0
        if r < 0.4:
            resting_egc = 1
        if r < 0.2:
            resting_egc = 2
        if r < 0.1:
            resting_egc = 3
        max_hr = random.randint(90, 202)
        excersice = 0 if random.random() > 0.55 else 0
        oldpeak = random.random() * 6.2
        r = random.random()
        st_slope = 0
        if r < 0.5:
            st_slope = 0
        if r < 0.3:
            st_slope = 1
        if r < 0.07:
            st_slope = 2
        heart_disease = 0 if random.random() > 0.5 else 1
        f.write(
            f"{age},{gender},{resting},{cholesterol},{fasting_rm},{resting_egc},{max_hr},{excersice},{oldpeak:.2f},{st_slope},{heart_disease}\n"
        )
