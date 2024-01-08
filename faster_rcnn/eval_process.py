import matplotlib.pyplot as plt

def read_log_file(file_path):
    maap50, map50_95 = [], []

    with open(file_path, 'r') as file:
        for line in file:
            elements = line.split()

            if elements[2] == 'Validation':
                if elements[3] == 'maAP50':
                    maap50.append(float(elements[-1]))
                elif elements[3] == 'mAP50-95:':
                    map50_95.append(float(elements[-1]))

    return maap50, map50_95

def plot_graph(values, title, ylabel):
    plt.figure(figsize=(4, 5))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel('Checkpoint Index')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

log_file_path = 'log 5class 3 unfrozen.txt'
maap50, map50_95 = read_log_file(log_file_path)

plot_graph(maap50, 'Validation mAP50 over time', 'mAP50')
plot_graph(map50_95, 'Validation mAP50-95 over time', 'mAP50-95')
