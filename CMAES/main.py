import numpy as np
import matplotlib.pyplot as plt


def levi_func(x1, x2):
    """
       https://en.wikipedia.org/wiki/Test_functions_for_optimization
    Args:
        -10, <= x1, x2  <= 10
        global minimum = f(1, 1) = 0
    """
    return 0.1 * (np.sin(3*np.pi*x1)**2
                  + (x1 - 1)**2 * (1 + np.sin(3 * np.pi * x2)**2)
                  + (x2 - 1)**2 * (1 + np.sin(2*np.pi * x2)**2))


def contor_plot(x1=None, x2=None):

    X1_list = np.linspace(-10, 10, 100)
    X2_list = np.linspace(-10, 10, 100)
    X1, X2 = np.meshgrid(X1_list, X2_list)
    Z = levi_func(X1, X2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    levels = [0, 0.05, 0.1, 0.5, 1, 3, 5, 10, 15, 20, 25, 30, 40]
    cp = ax.contourf(X1, X2, Z, levels=levels)
    ax.scatter([1], [1], marker="*", s=100, c="yellow")
    fig.colorbar(cp)
    return fig, ax


class CMAES:

    def __init__(self, centroid, sigma):

        #: 次元数
        self.dim = len(centroid)

        #: 世代ごと個体数
        self.lam = int(4 + 3*np.log(self.dim))

        #: 世代内エリート数
        self.mu = int(self.lam / 2)

        #: 正規分布の中心
        self.centoroid = centroid

        #: 分散共分散行列
        self.C = np.identity(self.dim)

        #: スケール
        self.sigma = sigma

        #: 進化パス
        self.p_c = np.zeros_like(self.C)

        self.p_sigma = np.zeros_like(self.sigma)

        weights = np.log(0.5*(self.lam + 1)) - np.log(np.arange(1, 1+self.mu))

        self.weights = weights / weights.sum()

        self.mueff = 1. / sum(self.weights ** 2)

        self.cc = 4. / (self.dim + 4.)

    def sample_population(self):
        """個体群の発生
            B = np.linalg.cholesky(self.C)でも実装できるが、
            updateのときに必要になるので面倒なやりかたで実装

            C = BDDB^T
        """
        #: z ~ N(0, 1)
        z = np.random.normal(0, 1, size=(self.lam, self.dim))

        #: C = B√D√DB.T
        diagD, B = np.linalg.eigh(self.C)
        BD = np.matmul(B, np.diag(np.sqrt(diagD)))

        #: y~N(0, C)
        y = np.matmul(BD, z.T).T
        #: X~N(μ, σC)
        X = self.centoroid + self.sigma * y

        individuals = [X[i] for i in range(X.shape[0])]

        return individuals

    def update(self, indiviuals, fitnesses):

        inds_with_fit = [(ind, fit) for ind, fit in zip(indiviuals, fitnesses)]
        inds_with_fit = sorted(inds_with_fit, key=(lambda ind: ind[1]))
        elites = inds_with_fit[:self.mu]

        import pdb; pdb.set_trace()


def main(n_generation=10):
    history = {}

    cmaes = CMAES(centroid=[-8, -8], sigma=0.5)

    fig, ax = contor_plot()
    for n in range(n_generation):
        individuals = cmaes.sample_population()
        fitnesses = [levi_func(*ind) for ind in individuals]
        cmaes.update(individuals, fitnesses)

        history[n] = individuals

        if n % 3 == 0:
            ax.scatter([ind[0] for ind in individuals],
                       [ind[1] for ind in individuals],
                       label=f"Gen: {n}", edgecolors="white")
        break

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
