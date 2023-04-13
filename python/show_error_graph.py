import matplotlib.pyplot as plt
from pyperclip import paste 

lines = paste().strip().split("\n")
numbers = [line.split(" ")[-1] for line in lines]
numbers = [float(num) for num in numbers]

plt.title("Training Error")
plt.xlabel("Epoch")
plt.ylabel("Error")

plt.plot(numbers)
plt.show()
