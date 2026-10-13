import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a simple salary dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [25000, 30000, 35000, 40000, 45000,
               52000, 58000, 65000, 72000, 80000]
}

df = pd.DataFrame(data)

# Select variables
X = df["Experience"].values
y = df["Salary"].values

# Linear Regression
slope, intercept = np.polyfit(X, y, 1)

# Predict salaries
predicted_salary = slope * X + intercept

# Calculate Mean Squared Error
mse = np.mean((y - predicted_salary) ** 2)

print("Salary Prediction Model")
print("-----------------------")
print("Slope:", round(slope, 2))
print("Intercept:", round(intercept, 2))
print("Mean Squared Error:", round(mse, 2))

# Predict salary for 6 years of experience
years = 6
prediction = slope * years + intercept

print("Predicted salary for 6 years experience:",
      round(prediction, 2))

# Create graph
plt.figure(figsize=(8, 6))

plt.scatter(X, y, label="Actual Salary")
plt.plot(X, predicted_salary, label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig("salary_prediction.png", dpi=300)

plt.show()

print("Salary Prediction project completed successfully!")