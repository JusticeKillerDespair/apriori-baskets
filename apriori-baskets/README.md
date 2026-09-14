# 基于 Apriori 算法的频繁项集挖掘

## 项目说明

本项目使用 Apriori 算法，在超市购物篮数据集 `baskets.csv` 中查找频繁项集。
程序会输出每个频繁项集及其支持度，并支持两种排序方式：

- 按支持度降序；
- 按字典序。

## 文件结构

```text
apriori-baskets/
├── baskets.csv
├── apriori.py
├── experiments.py
├── requirements.txt
├── README.md
└── results/
```

## 环境要求

- Python 3.10 或更高版本
- matplotlib

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行 Apriori 程序

示例：使用 5% 支持度，并按支持度降序输出。

```bash
python apriori.py --input baskets.csv --min-support 0.05 --sort support
```

示例：使用 5% 支持度，并按字典序输出。

```bash
python apriori.py --input baskets.csv --min-support 0.05 --sort lex
```

参数说明：

- `--input`：CSV 数据集路径；
- `--min-support`：最小支持度，例如 `0.05` 表示 5%；
- `--sort`：排序方式；
  - `support`：按支持度降序；
  - `lex`：按字典序。

## 运行实验

```bash
python experiments.py
```

实验会测试以下支持度阈值：

- 1%
- 3%
- 5%
- 10%
- 15%

实验结果保存在 `results/` 目录中：

```text
results/
├── experiment_results.csv
├── experiment_results.json
├── time_vs_support.png
├── lengths_vs_support.png
└── lengths_vs_support_bars.png
```

其中：

- `time_vs_support.png`：运行时间随支持度阈值变化图；
- `lengths_vs_support.png`：不同长度频繁项集数量随支持度阈值变化图；
- `lengths_vs_support_bars.png`：对应的柱状图；
- `experiment_results.csv`：实验数据表格。

## 数据集说明

`baskets.csv` 是超市购物篮数据。
每一行表示一次交易，商品之间用逗号分隔。
同一行中重复出现的商品只计算一次。

## 作者

王向杰