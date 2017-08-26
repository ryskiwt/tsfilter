## 想定環境
- python3系
- numpy, scipy
- numba (particle filterのみ)

## テストデータ生成

```py
import time, datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import random

from tsfilter import kalman, particle


random.seed(0)

# 系列生成
dtidx = pd.date_range('2017-01-01 00:00', periods=1000, freq='100L')
dtidx_sin = dtidx + datetime.timedelta(milliseconds=33)
dtidx_cos = dtidx - datetime.timedelta(milliseconds=33)
f = 0.1
th_sin = dtidx_sin.astype(np.int64).values * 2*np.pi *f / 1e9
th_cos = dtidx_cos.astype(np.int64).values * 2*np.pi *f / 1e9
sin = np.sin(th_sin) + 0.5 * np.sin(10*th_sin)  + 0.2*random.randn(*th_sin.shape) -2 + 4 * np.linspace(0, 1, len(th_sin))
cos = np.cos(th_cos) + 0.5 * np.sin(10*th_sin) + 0.2*random.randn(*th_cos.shape) -2 + 4 * np.linspace(0, 1, len(th_cos))
sin[20:80] = np.nan
sin[150:160] = np.nan
df = pd.DataFrame({
    'sin': pd.Series( sin, index=dtidx_sin ),
    'cos': pd.Series( cos, index=dtidx_cos ),
})

print("sample number: {0}".format(len(df)))
```

## カルマンフィルタ

```py
# 遷移モデル
F = np.kron(np.eye(2), np.array([[2, -1], [1, 0]])).astype(DTYPE)
G = np.kron(np.eye(2), np.array([[1, 0]])).astype(DTYPE).T
Q = 0.01 * np.eye(2, dtype=DTYPE)

# 観測モデル
H = np.kron(np.eye(2), np.array([[1, 0]])).astype(DTYPE)
R = 3 * np.eye(2, dtype=DTYPE)

# 初期値
x0 = np.r_[df['sin'].dropna().iloc[0], df['sin'].dropna().iloc[0]]
x0 = np.kron(x0.reshape(-1,1), np.ones([2,1])).astype(DTYPE)
V0 = G @ Q @ G.T

# フィルター生成
kf = kalman.Filter(F, G, Q, H, R)

# 対数尤度
kf.initialize(x0, V0)
start = time.time()
llh = kf.loglikelihood(df.values.T)
elapsed_time = (time.time() - start) * 1e3
print("elapsed_time: {0}".format(elapsed_time) + "[ms]")
print("loglikelihood: {0}".format(llh))

# プロット準備
_, axes = plt.subplots(2,1, sharex=True)
df.plot(subplots=True, ax=axes, style='k.')

# フィルタリング
kf.initialize(x0, V0)
start = time.time()
y = kf.filtering(df.values.T)
elapsed_time = (time.time() - start) * 1e3
print("elapsed_time: {0}".format(elapsed_time) + "[ms]")
pd.DataFrame(y.T, columns=df.columns, index=df.index).plot(subplots=True, ax=axes, style='b-')

# 固定区間スムージング
kf.initialize(x0, V0)
start = time.time()
y = kf.smoothing(df.values.T)
elapsed_time = (time.time() - start) * 1e3
print("elapsed_time: {0}".format(elapsed_time) + "[ms]")
pd.DataFrame(y.T, columns=df.columns, index=df.index).plot(subplots=True, ax=axes, style='r-')

# プロット
legends = ['original', 'filtered', 'smoothed (fixed-interval)']
for i,c in enumerate(df.columns):
    axes[i].legend(legends)
    axes[i].set_title(c)
plt.show()
```

## パーティクルフィルタ

```py
num = 4000

# 遷移モデル
F = np.kron(np.eye(2), np.array([[2, -1], [1, 0]])).astype(np.float64)
sysgma = 0.1 * np.ones([2,1]).astype(np.float64)
def update(pars):
    pars[0:4,:] = F @ pars[0:4,:]
    pars[[0,2],:] += normal_noise(sysgma, num)
    # pars[[0,2],:] += cauchy_noise(sysgma, num)

# 観測モデル
def observe(pars):
    return pars[[0,2],:]

idx_noangles = [0, 1]
obsgma = 2 * np.ones([2,1]).astype(np.float64)
idx_angles = []
beta = 1 * np.ones([1,1]).astype(np.float64)

def loglikelihood(y, pars):
    pars_obs = observe(pars)
    llh = np.zeros(pars.shape[1])

    # 角度以外の場合は、正規分布を適用する
    if 0<len(idx_noangles):
        e = pars_obs[idx_noangles,:] - y[idx_noangles,:]
        llh += np.nansum(normal_logpdf(obsgma.reshape(-1,1), e), axis=0)
        # llh += np.nansum(cauchy_logpdf(e, obsgma.reshape(-1,1)), axis=0)

    # 角度の場合は、フォン・ミーゼス分布を適用する
    if 0<len(idx_angles):
        e = pars_obs[idx_angles,:] - y[idx_angles,:]
        llh += np.nansum(vonmises_logpdf(beta.reshape(-1,1), e), axis=0)

    return llh

# 推定
def estimate(pars, weights):
    pars_obs = observe(pars)
    s = np.empty(pars_obs.shape[0])

    # 角度以外は算術平均
    s[idx_noangles] = np.sum(
        pars_obs[idx_noangles,:] * weights.reshape(1,-1),
        axis=1,
    )

    # 角度は角度統計の平均を利用
    s[idx_angles] = np.angle(
        np.sum(
            np.exp(1j*pars_obs[idx_angles,:]) * weights.reshape(1,-1),
            axis=1,
        )
    )

    return s.reshape(1,-1)

# 初期値
x0 = np.r_[df['sin'].dropna().iloc[0], df['cos'].dropna().iloc[0]]
x0 = np.kron(x0.reshape(-1,1), np.ones([2,1]))
pars0 = (0.1 * np.random.randn(4,num) + x0).astype(np.float64)

# フィルター生成
pf = particle.Filter(update, loglikelihood, estimate)

# 対数尤度
pf.initialize(pars0)
start = time.time()
llh = pf.loglikelihood(df.values.T)
elapsed_time = (time.time() - start) * 1e3
print("elapsed_time: {0}".format(elapsed_time) + "[ms]")
print("loglikelihood: {0}".format(llh))

# プロット準備
_, axes = plt.subplots(2,1, sharex=True)
df.plot(subplots=True, ax=axes, style='k.')

# フィルタリング
pf.initialize(pars0)
start = time.time()
y = pf.filtering(df.values.T)
elapsed_time = (time.time() - start) * 1e3
print("elapsed_time: {0}".format(elapsed_time) + "[ms]")
pd.DataFrame(y.T, columns=df.columns, index=df.index).plot(subplots=True, ax=axes, style='b-')

# 固定ラグ平滑化
start = time.time()
pf.initialize(pars0)
y = pf.smoothing(df.values.T)
elapsed_time = (time.time() - start) * 1e3
print("elapsed_time: {0}".format(elapsed_time) + "[ms]")
pd.DataFrame(y.T, columns=df.columns, index=df.index).plot(subplots=True, ax=axes, style='r-')

# プロット
legends = ['original', 'filtered', 'smoothed (fixed-interval)']
for i,c in enumerate(df.columns):
    axes[i].legend(legends)
    axes[i].set_title(c)
plt.show()
```
