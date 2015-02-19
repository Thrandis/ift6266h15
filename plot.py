import matplotlib.pylab as plt
import os

xp_path = 'xp'

def plot_results(result_file):
    out_dir  = os.path.dirname(result_file)
    epochs = []
    t_err = []
    v_err = []
    t_miss = []
    v_miss = []
    with open(result_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            epochs.append(float(line[0]))
            t_err.append(float(line[1]))
            t_miss.append(float(line[2]))
            v_err.append(float(line[3]))
            v_miss.append(float(line[4]))
    if max(t_miss) > 1:
        t_miss = [1 - x/100. for x in t_miss]
        v_miss = [1 - x/100. for x in v_miss]
    plt.figure()
    plt.plot(epochs, t_err, c='b', lw=2, label='Train')
    plt.grid()
    plt.plot(epochs, v_err, c='r', lw=2, label='Valid')
    plt.legend()
    plt.xlim(0, len(epochs))
    plt.xlabel('Epochs'), plt.ylabel('Criterion error')
    plt.title('Criterion Error')
    plt.savefig(os.path.join(out_dir, 'criterion_error.png'))
   
    plt.figure()
    plt.plot(epochs, t_miss, c='b', lw=2, label='Train')
    plt.grid()
    plt.plot(epochs, v_miss, c='r', lw=2, label='Valid')
    plt.legend()
    plt.xlim(0, len(epochs)), plt.ylim(0, 0.5)
    plt.xlabel('Epochs'), plt.ylabel('Missclassification')
    plt.title('Missclassification Rate')
    plt.savefig(os.path.join(out_dir, 'missclass_rate.png'))

xps = [os.path.join(xp_path, x) for x in os.listdir(xp_path) if x != 'logs']
for xp in xps:
    result_file = os.path.join(xp, 'results.txt')
    plot_results(result_file)
