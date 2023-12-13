import matplotlib.pyplot as plt
Loss_GCN = [6.18623, 6.14445, 6.11529, 6.09511, 6.07953, 6.00806, 6.00643, 6.00418, 6.00277, 6.00114, 5.99244, 5.98982, 5.98894, 5.98813, 5.98755, 5.98361, 5.98097, 5.98003, 5.97952, 5.97902, 5.97536, 5.9739, 5.97292, 5.973, 5.97213]
macro_f1_GCN = [0.7176, 0.7699, 0.7920, 0.8067, 0.8056, 0.8181, 0.8224, 0.8261, 0.8228, 0.8197, 0.8235, 0.8222, 0.8283, 0.8276, 0.8228, 0.8137, 0.8202, 0.8240, 0.8199, 0.8248, 0.8254, 0.8229, 0.8150, 0.8214, 0.8253]
micro_f1_GCN = [0.7540, 0.7948, 0.8122, 0.8232, 0.8215, 0.8309, 0.8339, 0.8358, 0.8353, 0.8329, 0.8360, 0.8337, 0.8377, 0.8356, 0.8327, 0.8228, 0.8296, 0.8320, 0.8278, 0.8316, 0.8339, 0.8311, 0.8249, 0.8303, 0.8303]

# Given data
losses = [
    6.22535, 6.17383, 6.14193, 6.12006, 6.10427, 6.02628, 6.02352,
    6.02245, 6.02002, 6.01924, 6.00561, 6.00407, 6.002, 6.00068,
    6.00026, 5.99357, 5.99196, 5.9899, 5.98882, 5.98884,
    5.98373, 5.98269, 5.98219, 5.98179, 5.9816
]
macro_f1 = [
    0.6797, 0.7566, 0.7828, 0.7956, 0.8009, 0.8081, 0.8100, 0.8133, 0.8139,
    0.8153, 0.8153, 0.8242, 0.8216, 0.8269, 0.8227, 0.8182, 0.8173, 0.8216,
    0.8175, 0.8186, 0.8202, 0.8226, 0.8226, 0.8200, 0.8226, 0.8185, 0.8202
]
micro_f1 = [
    0.7180, 0.7852, 0.8055, 0.8156, 0.8176, 0.8243, 0.8232, 0.8305, 0.8326,
    0.8336, 0.8326, 0.8385, 0.8363, 0.8399, 0.8349, 0.8330, 0.8385, 0.8355,
    0.8338, 0.8288, 0.8282, 0.8311, 0.8362, 0.8381, 0.8355, 0.8360
]


epoch_interval = 5
epoch_markers = [i for i in range(epoch_interval, len(losses) + 1, epoch_interval)]

# Plotting the training loss
plt.figure(figsize=(10, 7))
# plt.plot(losses, label='Multi-Head Loss', color='red', linestyle='--')

# plt.plot(Loss_GCN, label='No attention Loss', color='blue', linestyle='--')

plt.plot(range(len(micro_f1)), micro_f1, label='Multi-Head', color='red', marker='o')
plt.plot(range(len(micro_f1_GCN)), micro_f1_GCN, label='No Attention', color='blue', marker='o')

# Plotting the Macro F1 scores
# plt.plot(range(len(macro_f1)), macro_f1, label='Macro F1 Score', color='blue', marker='o')

# Plotting the Micro F1 scores
# plt.plot(range(len(micro_f1)), micro_f1, label='Micro F1 Score', color='green', marker='x')

# Adding titles and labels
# plt.title('Training Loss Over Epochs')
plt.title('Micro F1 Score Over Epochs')
plt.xlabel('Epochs')
# plt.ylabel('Loss')
plt.ylabel('Micro F1 Score')

plt.xticks(epoch_markers, labels=[str(i//epoch_interval) for i in epoch_markers])
# plt.xticks(range(len(losses)), range(1, len(losses) + 1))  # Assuming each value corresponds to an epoch

# Adding a legend
plt.legend()

# Display the plot
plt.show()
