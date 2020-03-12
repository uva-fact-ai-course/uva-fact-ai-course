import matplotlib as plt

class VGraph:
    '''Class specifically designed to kind of replace tensorboard as it does not
       work with my built and breaks python if I try to fix it.'''

    def __init__(self, title, xlab, ylab):
        self.title = title
        self.x = []
        self.y = []
        self.xlabel = xlab
        self.ylabel = ylab

    def add_point(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def show(self):
        plt.plot(self.x, self.y)
        plt.title(self.title)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        plt.show()

    def save(self, path, show=False):
        if not os.path.exists(path):
            os.makedirs(path)
        plt.plot(self.x, self.y)
        plt.title(self.title)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        plt.legend()
        plt.savefig(path)
        if show:
            plt.show()

    def save2(self, other, path, fname, title, show=False):
        '''Makes a plot together with the other VGraph '''
        if not os.path.exists(path):
            os.makedirs(path)
        plt.plot(self.x, self.y, label=self.title)
        plt.plot(other.x, other.y, label=other.title)
        plt.title(title)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        plt.legend()
        plt.savefig(f'{path}/{fname}')
        if show:
            plt.show()
