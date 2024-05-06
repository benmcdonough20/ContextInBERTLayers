import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x = np.array([6,8,10,12])
y_gender = np.array([0.635, 0.575, 0.67, 0.74]) #Measurements recorded by hand
y_proxy = np.array([0.545, 0.559, 0.712, 0.7925]) #""

sns.set_style("darkgrid")
plt.errorbar(x, y_proxy, yerr=[0.005,0.005,0.02,0.0058], color='red', label = "proximity")  # Plot the chart
plt.errorbar(x, y_gender, yerr=[0.01, 0.03, 0.02, 0.01], color='blue', label = "gender")  # Plot the chart
plt.xlabel("Layer")  # add X-axis label
plt.ylabel("Accuracy")  # add Y-axis label
plt.title("Gender and Proximity in Pronoun Reference Encoding")  # add title
plt.legend()
plt.savefig("gendervproxy.png", dpi=300)