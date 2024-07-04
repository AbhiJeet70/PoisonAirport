import matplotlib.pyplot as plt

# Function for plotting accuracy against perturbation percentages
def plot_accuracy_vs_perturbation(perturbation_percentages, accuracies, country):
    plt.figure(figsize=(10, 6))
    plt.plot([0] + [p * 100 for p in perturbation_percentages[1:]], accuracies, marker='o')
    plt.xlabel('Perturbation Percentage')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Perturbation for {country}')
    plt.grid(True)
    plt.show()
