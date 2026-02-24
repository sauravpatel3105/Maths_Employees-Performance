
# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import acos, degrees

# 2. Load Dataset
df = pd.read_csv("Employees Performance DATASET.csv")

print("Dataset Loaded Successfully")
print(df.head())
print("\n----------------------------------\n")

print("STEP 1: CENTRAL TENDENCY & DISPERSION\n")

mean_salary = df["Salary"].mean()
median_salary = df["Salary"].median()
mode_salary = df["Salary"].mode()[0]

variance_projects = df["Projects_Completed"].var()
std_projects = df["Projects_Completed"].std()

print(f"Mean Salary   : {mean_salary:.2f}")
print(f"Median Salary : {median_salary:.2f}")
print(f"Mode Salary   : {mode_salary:.2f}")
print(f"Variance (Projects Completed): {variance_projects:.2f}")
print(f"Standard Deviation (Projects Completed): {std_projects:.2f}")

print("\n----------------------------------\n")

print("STEP 2: PROBABILITY\n")

# Probability of Promotion
promotion_probability = (df["Promotion_Status"] == "Yes").mean()
print(f"Probability of Promotion: {promotion_probability:.2f}")

# Conditional Probability: Promotion | Performance_Score > 80
high_perf = df[df["Performance_Score"] > 80]
conditional_prob = (high_perf["Promotion_Status"] == "Yes").mean()

print("Conditional Probability (Promotion | Performance > 80):",
      f"{conditional_prob:.2f}")

print("\n----------------------------------\n")

print("STEP 3: DISTRIBUTIONS & VISUALIZATIONS\n")

plt.figure()
plt.hist(df["Performance_Score"], bins=30, density=True)
mean = df["Performance_Score"].mean()
std = df["Performance_Score"].std()

x = np.linspace(df["Performance_Score"].min(),
                df["Performance_Score"].max(), 100)
y = stats.norm.pdf(x, mean, std)
plt.plot(x, y)

plt.title("Performance Score Distribution with Gaussian Curve")
plt.xlabel("Performance Score")
plt.ylabel("Density")
plt.show()

salary_skewness = stats.skew(df["Salary"])
salary_kurtosis = stats.kurtosis(df["Salary"])

print(f"Salary Skewness : {salary_skewness:.2f}")
print(f"Salary Kurtosis : {salary_kurtosis:.2f}")

plt.figure()
stats.probplot(df["Projects_Completed"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Projects Completed")
plt.show()

print("\n----------------------------------\n")

print("STEP 4: LINEAR ALGEBRA\n")

v1 = df.loc[0, ["Projects_Completed", "Working_Hours"]].values
v2 = df.loc[1, ["Projects_Completed", "Working_Hours"]].values

dot_product = np.dot(v1, v2)

norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)

cos_theta = dot_product / (norm_v1 * norm_v2)
angle = degrees(acos(cos_theta))

print(f"Vector 1 (Employee 1): {v1}")
print(f"Vector 2 (Employee 2): {v2}")
print(f"Dot Product           : {dot_product:.2f}")
print(f"Norm of Vector 1      : {norm_v1:.2f}")
print(f"Norm of Vector 2      : {norm_v2:.2f}")
print(f"Angle Between Vectors : {angle:.2f} degrees")

print("\n----------------------------------\n")

print("KEY INSIGHTS\n")

print("1. Employees with Performance_Score > 80 have higher promotion probability.")
print("2. Salary distribution shows skewness, indicating unequal pay structure.")
print("3. Projects Completed and Working Hours show measurable work intensity.")
print("4. Linear algebra helps compare employee work patterns mathematically.")

print("\n--- COMPLETED ---")