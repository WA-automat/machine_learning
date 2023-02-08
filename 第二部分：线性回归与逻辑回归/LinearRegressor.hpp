#pragma once

#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<cstring>

using namespace std;
#define DEBUG

using LL = long long;

// 线性回归器

/// <summary>
/// 线性回归模型
/// 模板参数是自变量的维度
/// </summary>
template<unsigned N>
class LinearRegressor {

	// 公有成员
public:

	// 构造函数
	LinearRegressor() = default;
	
	// 析构函数
	~LinearRegressor() = default;

	// 训练模型的函数
	void fit(

		// 自变量与因变量
		vector<vector<double>> X, vector<double> y,

		// 最大迭代次数
		size_t iterations = INT_MAX,

		// 最大评估次数
		size_t evaluate = 10 * pow(2, N)

	) {

		// 非法情况
		size_t xlen = X.size(), ylen = y.size();
		if (xlen != ylen || (xlen == 1 && N == 1)) throw runtime_error("error");

		// 获取数据量
		size_t len = xlen;
		evaluate *= pow(2, len);

		// 标记，用于判断迭代是否完成
		bool flag = false;
		size_t times = 0, idx = 0;
		double ans = 0, errorCode = 0x3fffffff;

		// 随机数
		srand(static_cast<unsigned>(time(NULL)));

		// 查询最值
		double maxX = 0, minX = 0x3f3f3f3f, maxY = 0, minY = 0x3f3f3f3f;
		for (int i = 0; i < len; ++i) {
			for (int j = 0; j < N; ++j) {
				minX = min(minX, X[i][j]);
				maxX = max(maxX, X[i][j]);
			}
		}
		for (int i = 0; i < len; ++i) {
			maxY = max(maxY, y[i]);
			minY = min(minY, y[i]);
		}

		// 计算最大虚拟斜率
		double myMax = static_cast<double>(maxY - minY) / static_cast<double>(maxX - minX) * 2;

		// 随机生成w与b
		for (int i = 0; i < N; ++i) {
			w[i] = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;
		}
		b = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;

		// 保存最优结果的数组
		double tmpW[N];
		double tmpB = 0;

		memset(tmpW, 0, sizeof tmpW);

		// 开始迭代
		while (!flag && times < iterations) {

			// 计算当前解的误差
			ans = 0;

			// 遍历每一个元组
			for (int i = 0; i < len; ++i) {
				
				// 求当前预测值
				double fy = 0;
				// 计算每一个属性
				for (int j = 0; j < N; ++j) fy += w[j] * X[i][j];
				fy += b;

				// 损失函数计算
				ans += pow(fy - y[i], 2);

			}

			// 调试
//#ifdef DEBUG
//			cout << times << ":" << endl;
//			for (int i = 0; i < N; ++i) {
//				cout << "w[" << i << "] = " << w[i] << endl;
//			}
//			cout << "b = " << b << endl;
//			cout << endl;
//#endif // DEBUG

			// 判断迭代效果
			if (ans < errorCode) {
				
				idx = 0;
				errorCode = ans;

#ifdef DEBUG
				cout << times << ":" << endl;
				for (int i = 0; i < N; ++i) {
					cout << "w[" << i << "] = " << w[i] << endl;
				}
				cout << "b = " << b << endl;
				cout << endl;
#endif // DEBUG

				// 保存最优结果
				for (int i = 0; i < N; ++i) {
					tmpW[i] = w[i];
				}
				tmpB = b;

			}
			else {

				// 连续好几次找不到，说明已经大致收敛
				++idx;
				if (idx > evaluate) break;

			}

			if (idx < evaluate) {

				// 重新随机w与b
				for (int i = 0; i < N; ++i) {
					w[i] = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;
				}
				b = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;

			}
			// 迭代次数加一
			++times;

		}

		// 将最优结果保存下来
		for (int i = 0; i < N; ++i) w[i] = tmpW[i];
		b = tmpB;

	}

	// 线性回归预测函数
	double predict(vector<double> x) {
		
		// 计算结果
		double ans = 0;

		for (int i = 0; i < N; ++i) {
			ans += w[i] * x[i];
		}

		return ans + b;

	}

	// 私有成员
private:

	// 线性回归模型的参数
	double w[N];
	double b;

};

