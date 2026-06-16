import matplotlib.pyplot as plt


def plot_results(results, save_path="results.png"):

    N = [r[0] for r in results]
    acc = [r[1] for r in results]

    plt.figure(figsize=(7,5))

    plt.plot(N, acc, marker='o', linewidth=2)

    plt.title("Few-shot kNN with DINOv2 features (CIFAR-10)")
    plt.xlabel("N shots per class")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.show()