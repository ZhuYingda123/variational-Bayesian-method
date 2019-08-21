# 变分贝叶斯方法

## 	1 原理介绍

变分贝叶斯(VB)是一类用于贝叶斯估计和机器学习领域中近似计算复杂（intractable）积分的技术。它主要应用于复杂的统计模型中，这种模型一般包括三类变量：观测变量(observed  variables, data)，未知参数（parameters）和潜变量（latent  variables）。在贝叶斯推断中，参数和潜变量统称为不可观测变量(unobserved variables)。变分贝叶斯方法主要是两个目的:<br>

1. 近似不可观测变量的后验概率，以便通过这些变量作出统计推断。

2. 对一个特定的模型，给出观测变量的边缘似然函数 marginal probability（或称为证据，evidence）的下界。主要用于模型的选择，认为模型的边缘似然值越高，则模型对数据拟合程度越好，该模型产生Data的概率也越高。

变分贝叶斯方法可看做[EM][em网址]算法的扩展，[EM][em网址]算法是带有潜变量的极大似然估计(ML)， 其最终返回的是最有可能产生被观测数据的模型参数（在局部最优的意义下，并不一定是全局最优），而变分贝叶斯方法采用的是极大后验估计(MAP)，其返回的是模型参数的一个后验概率分布，而不是确定的一组值。<br>

假设观测数据由以下概率分布函数产生：
$$
p(X|\vec{\pi}, \vec{\theta})=\prod\limits_{i=1}^{N}(\sum\limits_{k=1}^{K}\pi_{k}p_k(\vec{x}_i|\vec{\theta}_k))
$$
这是一个生成模型，其中$\vec{\pi}=[\pi_1,...,\pi_K]$是混合概率向量，每个$\pi_k$表示成分$p_k(\vec{x}|\vec{\theta}_k)$的比重，$\vec{\theta}=[\vec{\theta}_1,...,\vec{\theta}_K]$表示模型参数。除了$\vec{\theta}$外，还有潜变量$Y = \{\vec{y}_1,..,\vec{y}_N\}$表示每个数据由哪个成分产生，其中每个$\vec{y}_i$是$K$维one-hot向量， 例如数据$\vec{x}_i$是由第$k$个成分产生，则$\vec{y}_i$的第$k$个分量为1,其余分量等于0。由于我们不知道每个观测数据是由哪个成分产生，因此$Y$是隐藏变量，在概率分布函数中也不能直接体现。<br>

若我们假设$Y$已知，则$X$的产生概率变成
$$
p(X|Y, \vec{\pi}, \vec{\theta})=\prod\limits_{i=1}^{N}\prod\limits_{k=1}^{K}p_k(\vec{x}_i|\vec{\theta}_k)^{y_{ik}}		
$$
在我们不知道$Y$的信息的情况下，$Y$也服从一个分布，这个分布显然和混合概率有关
$$
p(Y|\vec{\pi})=\prod\limits_{i=1}^{N}\prod\limits_{k=1}^{K}\pi^{y_{ik}}
$$
变分贝叶斯推断的目的就是通过观测到的$X$去估计参数$\vec{\pi}$， $\vec{\theta}$和潜变量$Y$的后验分布， 因此首先要给$\vec{\pi}$， $\vec{\theta}$假设一个先验分布，记为$p(\vec{\pi})$，$p(\vec{\theta}) $($Y$的先验分布就是式（2))， 要求的后验概率记为$p(Y,\vec{\pi},\vec{\theta}|X)$， 由于直接利用贝叶斯公式求后验概率涉及到不可行的积分计算[^1]， 因此需要用如下方法估计后验概率：<br>

1. 首先构造对数证据的下界，变分自由能$\mathcal{F}(q)$
   $$
   \log p(X) = \log \int p(X,Y,\vec{\pi},\vec{\theta})dYd\vec{\pi}d\vec{\theta}\\
      =\log \int p(X,Y,\vec{\pi},\vec{\theta})\frac{q(Y,\vec{\pi},\vec{\theta})}{q(Y,\vec{\pi},\vec{\theta})}dYd\vec{\pi}d\vec{\theta} \\
   \geq \mathbb{E}_q[\log p(X,Y,\vec{\pi},\vec{\theta})] - \mathbb{E}_q[\log q(Y,\vec{\pi}, \vec{\theta})] \\
   =\mathcal{F}(q)
   $$
其中$q(Y,\vec{\pi}, \vec{\theta})$是变分分布，我们将用变分分布去近似追踪后验分布，最后的不等式利用了 Jensen's 不等式。<br>
   
   要用$q(Y,\vec{\pi}, \vec{\theta})$去近似$p(Y,\vec{\pi},\vec{\theta}|X)$，我们就要最小化两者之间的KL散度，
   $$
   KL(q(Y,\vec{\pi}, \vec{\theta})|p(Y,\vec{\pi},\vec{\theta}|X)) =\\
   \int q(Y,\vec{\pi}, \vec{\theta})\log(\frac{q(Y,\vec{\pi},\vec{\theta})}{p(Y,\vec{\pi},\vec{\theta}|X)})dYd\vec{\pi}d\vec{\theta} \\
   = -\mathcal{F}(q)+\log p(X)
   $$
   可见最小化KL散度等价于最大化变分自由能$\mathcal{F}(q)$。
   
2. 利用平均场理论，假设$q(Y,\vec{\pi}, \vec{\theta})$可以分解成如下形式，
   $$
   q(Y,\vec{\pi},\vec{\theta})=q_{\vec{\pi}}(\vec{\pi})q_{\vec{\theta}}(\vec{\theta})\prod\limits_{i}q_{\vec{y}_{i}}(\vec{y}_{i})
   $$
   
3. 每次固定其他变量，对一个变量进行变分，可得到类似于EM算法的迭代优化过程，例如变分自由能可写为，
   $$
   \mathcal{F}(q) = \mathbb{E}_q[\log p(X,Y,\vec{\pi}, \vec{\theta})]-\mathbb{E}_q[\log (q_{\vec{\pi}}q_{\vec{\theta}}\prod\limits_{i}q_{\vec{y}_{i}})] \\
   =\mathbb{E}_q[\log p(\vec{\theta}|X,Y,\vec{\pi})]-\mathbb{E}_q[\log (q_{\vec{\theta}}(\vec{\theta})]+C_{Y, \vec{\pi}}\\
   = -KL(q_{\vec{\theta}}(\vec{\theta})||q_{\vec{\theta}}^{*}(\vec{\theta}))+C_{Y, \vec{\pi}}
   $$
   其中$C_{Y, \vec{\pi}}$是关于$Y$和$\vec{\pi}$的函数，$q_{\vec{\theta}}^{*}(\vec{\theta})$的定义如下，
   $$
   q_{\vec{\theta}}^{*}(\vec{\theta})= \frac{1}{C}\exp \mathbb{E}_{Y,\vec{\pi}}[\log p(\vec{\theta}| X,Y,\vec{\pi})]
   $$
   令$q_{\vec{\theta}}(\vec{\theta})=q_{\vec{\theta}}^{*}(\vec{\theta})$就完成了对$\vec{\theta}$的变分分布的一次更新，类似的，$Y$和$\vec{\pi}$的变分分布的更新公式如下，
   $$
   q_{\vec{y}_{i}}^*(q_{\vec{y}_{i}})= \frac{1}{C}\exp \mathbb{E}_{\vec{\pi},\vec{\theta}}[\log p(\vec{y}_{i}| \vec{x}_{i},\vec{\pi},\vec{\theta})]
   $$

   $$
   q_{\vec{\pi}}^*(q_{\vec{\pi}})= \frac{1}{C}\exp \mathbb{E}_{Y,\vec{\theta}}[\log p(\vec{\pi}| X, Y, \vec{\theta})]
   $$

   更新潜在变量$Y$的步骤类比EM算法中的E步，更新参数$\vec{\pi}$和$\vec{\theta}$的步骤类比EM算法中的M步。
## 	2 优势

因为VB方法是EM算法的一种推广，这里仅讨论其相对于EM算法的优势。<br>

1.  VB算法克服了EM算法中存在的奇点问题。在EM算法中，当遇到某个成分只包含一个训练数据时，该成分的概率密度函数会出现奇点，例如假设该成分都服从高斯分布，则其方差会被训练到0，从而产生奇点，导致计算机无法处理。而在VB方法中，由于优化的是参数的后验分布，而非确定的极大似然值，就可以避免奇点的产生。

2. 我们将变分自由能重写为以下形式，
   $$
   \mathcal{F}(q) = \mathbb{E}_{Y,\vec{\pi},\vec{\theta}}[\log\frac{p(X,Y|\vec{\pi},\vec{\theta})}{q_Y(Y)}]-KL(q_{\vec{\pi}}(\vec{\pi})q_{\vec{\theta}}(\vec{\theta})||p(\vec{\pi})p(\vec{\theta}))
   $$
   右边第一项是对数似然的期望，第二项是参数的变分分布与先验分布之间的KL散度，可看作对结构风险的惩罚项。参数越多，惩罚就越大，因此VB算法倾向于选择简单的模型，因此相比于EM算法可以更好地避免过拟合。

## 	3 混合高斯模型下的VB

在混合高斯模型下，假设观测数据按以下条件概率产生
$$
p(X|Y, \vec{\mu}, \vec{\varLambda})=\prod\limits_{i=1}^{N}\prod\limits_{k=1}^{K}\mathcal{N}(\vec{x}_{i}|\vec{\mu}_k, \vec{\varLambda}_k^{-1})^{y_{ik}}
$$
其中$\mathcal{N}()$表示高斯分布函数，$\vec{\mu},\vec{\varLambda}$分别表示均值和方差，模型参数为$\vec{\theta}=[\vec{\mu},\vec{\varLambda}]$。<br>

参数的先验分布应满足[共轭条件][共轭网址]，隐变量$Y$和参数$\vec{\pi}$，$\vec{\theta}$的先验概率分布选择如下：
$$
p(Y|\vec{\pi})=\prod\limits_{i=1}^{N}\prod\limits_{k=1}^{K}\pi_k^{y_{ik}}\\
p(\vec{\pi})=\frac{\Gamma(K\alpha_0)}{\Gamma(\alpha_0)^K}\prod\limits_{k=1}^{K}\pi_k^{\alpha_0-1}\\
p(\vec{\mu}|\vec{\varLambda})=\prod\limits_{k=1}^{K}\mathcal{N}(\vec{\mu}_k;\vec{\mu}_0, (\beta_0\vec{\varLambda}_k)^{-1})\\
p(\vec{\varLambda})=\prod\limits_{k=1}^{K}\mathcal{W}(\vec{\varLambda}_k;\vec{W}_0, \nu_0)
$$
其中$\alpha_0,\beta_0,\vec{\mu}_0,\nu_0,\vec{W}_0$是固定的超参数，于是对后验分布的估计就转化成了对超参数的估计。在共轭先验的条件下，变分分布的形式应与先验分布相同，假设变分分布由下式确定
$$
q(Y|\vec{\pi})=\prod\limits_{i=1}^{N}\prod\limits_{k=1}^{K}r_{ik}^{y_{ik}}\\
q(\vec{\pi})=\frac{\Gamma(K\alpha_k)}{\Gamma(\alpha_k)^K}\prod\limits_{k=1}^{K}\pi_k^{\alpha_k-1}\\
q(\vec{\mu}|\vec{\varLambda})=\prod\limits_{k=1}^{K}\mathcal{N}(\vec{\mu}_k;\vec{m}_k, (\beta_k\vec{\varLambda}_k)^{-1})\\
q(\vec{\varLambda})=\prod\limits_{k=1}^{K}\mathcal{W}(\vec{\varLambda}_k;\vec{W}_k, \nu_k)
$$
经过计算，可得其超参数的更新公式如下(具体推导可参考wiki百科[Variational Bayesian methods][VB网址])
$$
\alpha_{k}=\alpha_0+N_{k}\\
\beta_{k}=\beta_0+N_{k}\\
\vec{m}_{k}=\frac{1}{\beta_{jk}}(\beta_0\vec{\mu}_0+N_k\vec{x}_{k})\\
\vec{W}_{k}^{-1}=\vec{W}_0^{-1}+\beta_0\vec{\mu}_0\vec{\mu}_0^T
	-\beta_{k}\vec{m}_{k}\vec{m}_{k}^T+\sum\limits_{i=1}^{N}r_{ik}\vec{x}_{i}\vec{x}_{i}^T\\
	\nu_{k}=\nu_0+N_{k}\\
	\vec{x}_{k}=\frac{1}{N_{k}}\sum\limits_{i=1}^{N}r_{ik}\vec{x}_{i}\\
	N_{k}=\sum\limits_{i=1}^{N}r_{ik}
$$
仔细考察可以发现，参数$r_{ik}$可以理解为第i个观测值$\vec{x}_i$属于第k个成分的概率，其计算公式见[Variational Bayesian methods][VB网址]。

[em网址] :  https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwid87CEiJHkAhWDQN4KHdwOAMcQFjAAegQIABAB&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FExpectation%25E2%2580%2593maximization_algorithm&usg=AOvVaw2gGiMhgx4ISRmXSmW6DkCt

[共轭网址] : https://zlearning.netlify.com/computer/prml/prmlch2-conjugate-priors

[VB网址] : https://en.wikipedia.org/wiki/Variational_Bayesian_methods

[^1]:  Hoffman M D, Blei D M, Wang C, et al. Stochastic variational inference[J]. The Journal of Machine Learning Research, 2013, 14(1): 1303-1347. 

