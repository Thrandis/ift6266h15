import matplotlib.pylab as plt


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
            epochs.append(line[0])
            t_err.append(line[1])
            t_miss.append(line[2])
            v_err.append(line[3])
            v_miss.append(line[4])
    plt.figure()
    plt.plot(epochs, t_err)
    plt.hold()
    plt.plot(epochs, v_err)
    plt.xlabel('Epochs'), plt.ylabel('Criterion error')
    plt.title('Criterion Error')
    plt.savefig(os.path.join(out_dir, 'criterion_error.png'))
    plt.figure()
    plt.plot(epochs, t_miss)
    plt.hold()
    plt.plot(epochs, v_miss)
    plt.xlabel('Epochs'), plt.ylabel('Missclassification')
    plt.title('Missclassification Rate')
    plt.savefig(os.path.join(out_dir, 'missclass_rate.png'))
        

