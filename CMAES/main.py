import functools

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
                  + 0.4 * (x2 - 1)**2 * (1 + np.sin(2*np.pi * x2)**2))


def contor_plot(x1=None, x2=None):

    X1_list = np.linspace(-12, 12, 100)
    X2_list = np.linspace(-12, 12, 100)
    X1, X2 = np.meshgrid(X1_list, X2_list)
    Z = levi_func(X1, X2)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    levels = [0, 0.05, 0.1, 0.5, 1, 3, 5, 10, 15, 20, 25, 30, 40]
    cp = ax.contourf(X1, X2, Z, levels=levels)
    ax.scatter([1], [1], marker="*", s=100, c="yellow")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    fig.colorbar(cp)
    return fig, ax


class CMAES:

    def __init__(self, centroid, sigma, lam=None):

        #: 次元数
        self.dim = len(centroid)

        #: 世代ごと個体数
        self.lam = lam if lam else int(4 + 3*np.log(self.dim))

        #: 世代内エリート数(親の数)
        self.mu = int(self.lam / 2)

        #: 正規分布中心
        self.centroid = centroid

        #: 順位にもとづく重み
        weights = np.log(0.5*(self.lam + 1)) - np.log(np.arange(1, 1+self.mu).reshape(1,-1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1. / (self.weights ** 2).sum()

        #: ステップサイズ： 進化パスと学習率
        self.sigma = sigma
        self.p_sigma = np.zeros(self.dim)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(
            0, np.sqrt((self.mu_eff - 1)/(self.dim + 1)) - 1
            ) + self.c_sigma

        #: 共分散行列： 進化パスと学習率
        self.C = np.identity(self.dim)
        self.p_c = np.zeros_like(self.C)
        self.c_c = 0



    def sample_population(self):
        """個体群の発生
            B = np.linalg.cholesky(self.C)でも実装できるが、
            updateのときに必要になるので面倒なやりかたで実装

            C = BDDB^T
        """
        #: z ~ N(0, 1)
        Z = np.random.normal(0, 1, size=(self.lam, self.dim))

        #: C = B√D√DB.T
        diagD, B = np.linalg.eigh(self.C)
        diagD = np.sqrt(diagD)
        BD = np.matmul(B, np.diag(diagD))

        #: y~N(0, C)
        Y = np.matmul(BD, Z.T).T
        #: X~N(μ, σC)
        X = self.centroid + self.sigma * Y

        return Z, Y, X

    def update(self, Z, Y, X, fitnesses, cma=True):

        """
            1. Selection and recombinatio
            上位μ個体の選抜と正規分布中心(centroid)の移動
        """
        #: あとで使うので現在の正規分布中心を保存
        old_centroid = self.centroid
        old_sigma = self.sigma

        #: エリートは上位μ個体(全個体数はλ)
        elite_indices = np.argsort(fitnesses)[:self.mu]
        X_elite = X[elite_indices, :]
        X_w = np.matmul(self.weights, X_elite)[0]
        self.centroid = X_w

        #: Note. Y_w = np.matmul(self.weights, Y[elite_indices, :])[0] でも可
        Y_w = (X_w - old_centroid) / old_sigma

        """
            2. Step-size control
            ステップサイズσの更新
            Note:
        """

        #: 対角行列の逆関数は対角成分の逆数をとればよい
        diagD, B = np.linalg.eigh(self.C)
        diagD = np.sqrt(diagD)
        inv_diagD = 1.0 / diagD

        #: Note. 定義からnp.matmul(B, Z.T).T == np.matmul(C_, Y.T).T
        C_ = np.matmul(np.matmul(B, np.diag(inv_diagD)), B.T)

        new_p_sigma = (1 - self.c_sigma) * self.p_sigma
        new_p_sigma += np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * np.matmul(C_, Y_w)
        self.p_sigma = new_p_sigma

        E_normal = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21 * self.dim **2))
        self.sigma = self.sigma * np.exp(
            (self.c_sigma / self.d_sigma)
            * (np.sqrt((self.p_sigma ** 2).sum()) / E_normal - 1)
        )


        """
            3. Covariance matrix adaptatio (CMA)
            共分散行列Cの更新
        """
        pass


def main(n_generation=30):

    cmaes = CMAES(centroid=[-10, -10], sigma=0.5, lam=12)

    history = {}
    fig, ax = contor_plot()
    for n in range(n_generation):
        Z, Y, X = cmaes.sample_population()
        fitnesses = levi_func(X[:, 0], X[:, 1])
        cmaes.update(Z, Y, X, fitnesses)

        history[n] = X
        if n % 5 == 0:
            ax.scatter(X[:, 0], X[:, 1],
                       label=f"Gen: {n}", edgecolors="white")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
