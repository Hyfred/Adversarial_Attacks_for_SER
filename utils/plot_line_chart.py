import matplotlib.pyplot as plt

# create dictionary with data points
data = {0:0.750, 0.02:0.699, 0.04:0.640, 0.06:0.587, 0.08:0.541, 0.1:0.523}

# extract x and y values
x_values = list(data.keys())
y_values = list(data.values())

# create a line chart
plt.plot(x_values, y_values, marker='o', label='VGG-16')

# set chart title and labels for axes
plt.title("Line Chart of Macc vs. Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Macc")
plt.legend()

# display the chart
# plt.show()
plt.savefig('line_chart.png')