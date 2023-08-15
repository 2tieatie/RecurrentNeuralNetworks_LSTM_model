from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self):
        pass

    def plot_results(self, real, coeff_predicts):
        plt.figure(figsize=(10, 6))
        plt.plot(real.index, real[['coefficient']], label='Real Coefficient')
        plt.plot(real.index, coeff_predicts, label='Predicted Coefficient', linestyle='dashed')
        plt.xlabel('Date')
        plt.ylabel('Coefficient')
        plt.title('Real vs Predicted Coefficient')
        plt.legend()
        plt.show()
