# numopt4j AI 使用指南

> 本文件内嵌于 JAR 包中，供 AI 编码助手快速理解 numopt4j 的使用方式。

## 1. 库功能概述

numopt4j 是一个高性能 Java 数值优化库，提供三种主流优化算法：

- **L-BFGS-B**：有界拟牛顿法，适用于无约束或边界约束优化
- **SLSQP**：序列最小二乘规划，适用于等式/不等式约束优化
- **TRF**：信赖域反射法，适用于有界非线性最小二乘拟合

所有算法均通过流式 API 调用，支持 lambda 表达式，无需手动计算梯度。

## 2. 算法选择指南

| 问题类型 | 推荐 API | 说明 |
|---------|---------|------|
| 无约束优化 | `LBFGSBProblem` 或 `Minimize.objective` | 最快，仅需目标函数 |
| 边界约束优化 | `LBFGSBProblem` + `.bounds()` | 支持 l ≤ x ≤ u |
| 等式约束优化 | `SLSQPProblem` + `.equalityConstraints()` | c(x) = 0 |
| 不等式约束优化 | `SLSQPProblem` + `.inequalityConstraints()` | c(x) ≥ 0 |
| 非线性最小二乘 | `TRFProblem` + `.residuals()` | min ½‖f(x)‖²，需 m ≥ n |
| 不确定 | `Minimize`（自动选择） | 根据问题特征自动选择算法 |


## 3. 核心 API 使用示例

### 最简用法（推荐 AI 生成代码使用）

```java
import com.curioloop.numopt4j.optim.*;
import java.util.Arrays;

// 无约束优化：只需提供目标函数，无需梯度
OptimizationResult r = Minimize.objective(x -> x[0]*x[0] + x[1]*x[1])
    .startingFrom(1.0, 1.0)
    .run();

if (r.isSuccessful()) {
    System.out.println(Arrays.toString(r.getSolution())); // [0.0, 0.0]
} else {
    System.out.println(r.getErrorMessage());
}
```

### LBFGSBProblem（无约束/边界约束）

```java
// 无约束优化
OptimizationResult r1 = new LBFGSBProblem()
    .objective(x -> x[0]*x[0] + x[1]*x[1])  // 自动使用 NumericalGradient.CENTRAL
    .initialPoint(1.0, 1.0)
    .solve();

// 边界约束优化
OptimizationResult r2 = new LBFGSBProblem()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .bounds(Bound.atLeast(0.5), Bound.atLeast(0.5))
    .initialPoint(1.0, 1.0)
    .maxIterations(200)
    .solve();

// 解析梯度（性能最优）
OptimizationResult r3 = new LBFGSBProblem()
    .objective((x, g) -> {
        double f = x[0]*x[0] + x[1]*x[1];
        if (g != null) { g[0] = 2*x[0]; g[1] = 2*x[1]; }
        return f;
    })
    .initialPoint(1.0, 1.0)
    .solve();
```

### SLSQPProblem（等式/不等式约束）

```java
// 等式约束：minimize x[0]^2 + x[1]^2, subject to x[0] + x[1] = 1
OptimizationResult r = new SLSQPProblem()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .equalityConstraints(x -> x[0] + x[1] - 1)
    .initialPoint(0.5, 0.5)
    .solve();
// solution: [0.5, 0.5]

// 不等式约束：minimize x[0] + x[1], subject to x[0]^2 + x[1]^2 >= 1
OptimizationResult r2 = new SLSQPProblem()
    .objective(x -> x[0] + x[1])
    .inequalityConstraints(x -> x[0]*x[0] + x[1]*x[1] - 1)
    .initialPoint(1.0, 1.0)
    .solve();
```

### TRFProblem（非线性最小二乘）

```java
// 拟合 y = a * exp(-b * t)
double[] tData = {0.0, 1.0, 2.0, 3.0};
double[] yData = {2.0, 1.2, 0.7, 0.4};

OptimizationResult r = new TRFProblem()
    .residuals((x, res) -> {
        for (int i = 0; i < tData.length; i++) {
            res[i] = yData[i] - x[0] * Math.exp(-x[1] * tData[i]);
        }
    }, tData.length)
    .bounds(Bound.atLeast(0), Bound.atLeast(0))
    .initialPoint(1.0, 0.5)
    .solve();

double a = r.getSolution()[0];
double b = r.getSolution()[1];
```


## 4. 常见错误及解决方案

### OptimizationException 错误码

| 错误码 | 原因 | 解决方案 |
|--------|------|---------|
| `MISSING_PARAM` | 未调用 `.objective()` 或 `.initialPoint()` | 在 `.solve()` 前设置所有必需参数 |
| `INVALID_INPUT` | 初始点含 NaN 或 Infinity | 确保所有初始值为有限数 |
| `DIMENSION_MISMATCH` | 初始点长度与期望维度不符 | 检查 `initialPoint` 数组长度 |
| `INVALID_VALUE` | 参数值不满足约束（如负数容差） | 参考错误消息中的约束说明 |

### 即时参数校验（IllegalArgumentException）

```java
// 以下调用会立即抛出 IllegalArgumentException
problem.objective(null);          // "objective function must not be null"
problem.initialPoint();           // "initialPoint must not be null or empty"
problem.maxIterations(0);         // "maxIterations must be positive, got 0"
problem.gradientTolerance(-1e-6); // "gradientTolerance must be positive, got -1.0E-6"
```

### 优化未收敛

```java
OptimizationResult r = new LBFGSBProblem()
    .objective(x -> x[0]*x[0])
    .initialPoint(1.0)
    .maxIterations(5)
    .solve();

if (!r.isSuccessful()) {
    System.out.println(r.toString());
    // OptimizationResult {
    //   status: Maximum iterations reached without convergence
    //   objectiveValue: ...
    //   iterations: 5
    //   evaluations: ...
    //   suggestion: Consider increasing maxIterations or relaxing tolerances
    // }

    // 根据建议调整参数
    r = new LBFGSBProblem()
        .objective(x -> x[0]*x[0])
        .initialPoint(1.0)
        .maxIterations(100)  // 增加迭代次数
        .solve();
}
```

### TRF m < n 错误

```java
// 残差数量必须 >= 参数数量
new TRFProblem()
    .residuals((x, r) -> { r[0] = x[0] - 1; r[1] = x[1] - 2; }, 2)  // m=2
    .initialPoint(0.0, 0.0)  // n=2，满足 m >= n
    .solve();
```

## 5. 性能提示

### 工作空间复用

在循环中多次求解相同维度的问题时，预分配工作空间可避免重复内存分配：

```java
LBFGSBProblem problem = new LBFGSBProblem()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .initialPoint(1.0, 1.0);

LBFGSBWorkspace ws = problem.alloc();  // 分配一次

for (double[] startPoint : startPoints) {
    OptimizationResult r = problem.initialPoint(startPoint).solve(ws);
    // 处理结果
}
```

### 解析梯度

当目标函数可以手动推导梯度时，使用 `Univariate` 接口比数值梯度快 2-4 倍：

```java
// 数值梯度（简单，每维度 2 次额外评估）
new LBFGSBProblem()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .initialPoint(1.0, 1.0)
    .solve();

// 解析梯度（更快，无额外评估）
new LBFGSBProblem()
    .objective((x, g) -> {
        double f = x[0]*x[0] + x[1]*x[1];
        if (g != null) { g[0] = 2*x[0]; g[1] = 2*x[1]; }
        return f;
    })
    .initialPoint(1.0, 1.0)
    .solve();
```

### 梯度方法选择

```java
// 默认（推荐）：中心差分，精度 O(h²) ≈ 1e-11
problem.objective(x -> f(x));  // 自动使用 NumericalGradient.CENTRAL

// 最快：前向差分，精度 O(h) ≈ 1e-8（函数评估昂贵时使用）
problem.objective(NumericalGradient.FORWARD, x -> f(x));

// 最精确：五点差分，精度 O(h⁴) ≈ 1e-15（需要极高精度时使用）
problem.objective(NumericalGradient.FIVE_POINT, x -> f(x));
```

---

完整 API 文档请参阅项目根目录的 `llms-full.txt` 或访问
[GitHub Wiki](https://github.com/curioloop/numopt4j/wiki/)。
