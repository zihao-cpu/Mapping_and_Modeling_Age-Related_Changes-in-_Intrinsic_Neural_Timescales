# Kinouchi-Copelli 模型代码详解与数学原理


这段代码实现了基于 **Kinouchi-Copelli（KC）模型** 的神经元网络模型，主要用于模拟神经元的激活动态、传播行为以及整体网络的活跃程度。本模型结合了马尔科夫过程、概率论和蒙特卡洛模拟等理论，具有重要的复杂系统研究意义。

---

## 1. `external()`：外部刺激函数

```python
@jit(nopython = True)
def external(t_min, t_max, t, r):
    if t_min < t < t_max:
        return (1 - np.exp(-r))
    else:
        return 0
```

### 数学原理：
外部刺激是一个随时间变化的激活因子，只有在时间区间 \( t_{\text{min}} < t < t_{\text{max}} \) 内生效。其强度随刺激强度参数 \( r \) 指数递增。

$$S_{\text{ext}}(t) = \begin{cases}1 - e^{-r}, & t_{\text{min}} < t < t_{\text{max}} \\0, & \text{otherwise}\end{cases}$$

---

## 2. `internal()`：内部神经刺激函数

```python
@jit(nopython = True)
def internal(p, index_matrix, index, state_matrix, degree):
    probability = 0
    for i in range(int(degree[index])):
        if (state_matrix[int(index_matrix[index][i])] == 1):
            probability = 1 - (1 - p[index][i]) * (1 - probability)
    return probability
```

### 数学原理：
该函数计算神经元 `index` 受到其连接邻居的激活概率。每个连接神经元单独激活是独立事件，总体激活概率是这些独立事件的联合补集的补集。


$$P_{\text{internal}} = 1 - \prod_{i} \left( 1 - p_{i} \cdot \delta_{\text{active}, i} \right)$$


其中，$$ \delta_{\text{active}, i} = 1$$ 表示神经元 $$ i $$ 当前为激活状态。

---

## 3. `state_update()`：状态更新函数

```python
@jit(nopython = True)
def state_update(r, t, t_min, t_max, p, index_matrix, index, 
                 state_matrix, degree, state_number, temp, random_arr):
    probability = 1 - ((1 - external(t_min, t_max, t, r)) * 
                        (1 - internal(p, index_matrix, index, state_matrix, degree)))
    if random_arr[index] < probability:
        temp[index] = 1
```

### 数学原理：
一个神经元的激活概率由**外部刺激**与**内部刺激**共同决定，它们是两个独立事件，其联合概率为：

$$P_{\text{activate}} = 1 - \left(1 - P_{\text{ext}}\right) \cdot \left(1 - P_{\text{internal}}\right)$$

如果随机数小于该概率，则该神经元激活。

---

## 4. `rho_determine()`：活跃神经元比例计算

```python
@jit(nopython = True)
def rho_determine(state_matrix):
    active = [i for i, v in enumerate(state_matrix) if v == 1]
    return len(active) / state_matrix.size
```

### 数学原理：
该函数计算当前网络中处于激活状态的神经元所占的比例：

$$\rho(t) = \frac{\text{Number of active neurons}}{\text{Total number of neurons}} = \frac{N_{\text{active}}}{N}$$

这是一种**宏观观测量**，可用于评估整体活动水平。

---

## 5. `KC_model()`：主模型函数

```python
@jit(nopython = True)
def KC_model(degree, index_matrix, p, t_min, stimulas_t, stimulas_S, MC_times, state_number):
    rho = np.zeros(MC_times)
    N = degree.size
    state_matrix = np.random.randint(0, state_number, N)
    temp = np.copy(state_matrix)

    for t in range(MC_times):
        temp[np.where(state_matrix == 0)[0]] -= 1 
        temp[np.where(state_matrix < (state_number - 1))] += 1 
        temp[np.where(state_matrix == (state_number - 1))] = 0 
        
        zeros = np.where(state_matrix == 0)[0]
        random_arr = np.random.random(N)
        for i in range(zeros.size):
            state_update(stimulas_S, t, t_min, stimulas_t, p, index_matrix, 
                         zeros[i], state_matrix, degree, state_number, temp, random_arr)
        state_matrix = np.copy(temp)
        rho[t] = rho_determine(state_matrix)
    return rho
```

### 数学原理：
该函数使用**蒙特卡洛模拟（Monte Carlo Simulation）**，对整个神经元网络进行时间演化模拟。每个时间步：

1. 更新恢复状态（refractory dynamics）；
2. 找出当前可激活神经元（状态为0）；
3. 对每个可激活神经元计算其激活概率；
4. 使用随机变量决定是否激活；
5. 记录当前的活跃度。

整个过程满足**马尔科夫性质**，即系统状态转移仅依赖当前状态：


$$P(X_{t+1} | X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} | X_t)$$

---

## 总结与理论基础

| 数学原理        | 应用点                                      |
| ----------- | ---------------------------------------- |
| **马尔科夫链**   | 状态更新只依赖当前状态                              |
| **概率论**     | 神经元激活是随机事件                               |
| **联合概率**    | 外部与内部刺激独立，使用 \( 1 - (1 - a)(1 - b) \) 组合 |
| **蒙特卡洛模拟**  | 用随机数推进系统演化                               |
| **图论**      | 网络结构通过 `index_matrix` 与 `degree` 建模      |
| **复杂系统临界性** | 可用于研究亚临界 / 临界 / 超临界状态下的网络行为              |

该模型具备探索**传播现象、自组织临界性（SOC）、信息处理极限**等广泛应用场景。
