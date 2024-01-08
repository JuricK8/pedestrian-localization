import matplotlib.pyplot as plt
import re

float_pattern = r"[-+]?\d*\.\d+|\d+"
def extract_loss_from_log(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()
        train = [[],[],[],[]]
        val = [[],[],[],[]]
        for i in range(0, len(lines)-1):
            line1 = lines[i].strip()[len("22024-01-08 00:41:36"):].strip()
            line2 = lines[i + 1].strip()[len("22024-01-08 00:41:36"):].strip()  

            if line1.startswith('{') and line2.startswith('{'):
                floats1 = re.findall(float_pattern, line1)
                float_values1 = [float(value) for value in floats1][::2]

                floats2 = re.findall(float_pattern, line2)
                float_values2 = [float(value) for value in floats2][::2]
                
                for cnt,f in enumerate(float_values1):
                    train[cnt].append(f / 2608)

                for cnt,f in enumerate(float_values2):
                    val[cnt].append(f / 329)


    return train, val

def plot_values(values, title, ylabel):
    plt.figure(figsize=(4, 5))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel('Pair Index')
    plt.ylabel(ylabel)

    file_name = title.replace(' ', '_').lower()  
    plt.savefig(f"{file_name}.png")

    plt.show()

log_file_path = 'train_log 1class 2unfrozen.txt'  
train, val = extract_loss_from_log(log_file_path)

plot_values(train[0] , 'Train Loss Classifier', 'Loss Classifier')
plot_values(train[1], 'Train Loss Box Reg', 'Loss Box Reg')
plot_values(train[2], 'Train Loss Objectness', 'Loss Objectness')
plot_values(train[3], 'Train Loss RPN Box Reg', 'Loss RPN Box Reg')


plot_values(val[0], 'Validation Loss Classifier', 'Loss Classifier')
plot_values(val[1], 'Validation Loss Box Reg', 'Loss Box Reg')
plot_values(val[2], 'Validation Loss Objectness', 'Loss Objectness')
plot_values(val[3], 'Validation Loss RPN Box Reg', 'Loss RPN Box Reg')
