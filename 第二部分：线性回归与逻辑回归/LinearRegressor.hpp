#pragma once

#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include<cstring>

using namespace std;
#define DEBUG

using LL = long long;

// ���Իع���

/// <summary>
/// ���Իع�ģ��
/// ģ��������Ա�����ά��
/// </summary>
template<unsigned N>
class LinearRegressor {

	// ���г�Ա
public:

	// ���캯��
	LinearRegressor() = default;
	
	// ��������
	~LinearRegressor() = default;

	// ѵ��ģ�͵ĺ���
	void fit(

		// �Ա����������
		vector<vector<double>> X, vector<double> y,

		// ����������
		size_t iterations = INT_MAX,

		// �����������
		size_t evaluate = 10 * pow(2, N)

	) {

		// �Ƿ����
		size_t xlen = X.size(), ylen = y.size();
		if (xlen != ylen || (xlen == 1 && N == 1)) throw runtime_error("error");

		// ��ȡ������
		size_t len = xlen;
		evaluate *= pow(2, len);

		// ��ǣ������жϵ����Ƿ����
		bool flag = false;
		size_t times = 0, idx = 0;
		double ans = 0, errorCode = 0x3fffffff;

		// �����
		srand(static_cast<unsigned>(time(NULL)));

		// ��ѯ��ֵ
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

		// �����������б��
		double myMax = static_cast<double>(maxY - minY) / static_cast<double>(maxX - minX) * 2;

		// �������w��b
		for (int i = 0; i < N; ++i) {
			w[i] = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;
		}
		b = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;

		// �������Ž��������
		double tmpW[N];
		double tmpB = 0;

		memset(tmpW, 0, sizeof tmpW);

		// ��ʼ����
		while (!flag && times < iterations) {

			// ���㵱ǰ������
			ans = 0;

			// ����ÿһ��Ԫ��
			for (int i = 0; i < len; ++i) {
				
				// ��ǰԤ��ֵ
				double fy = 0;
				// ����ÿһ������
				for (int j = 0; j < N; ++j) fy += w[j] * X[i][j];
				fy += b;

				// ��ʧ��������
				ans += pow(fy - y[i], 2);

			}

			// ����
//#ifdef DEBUG
//			cout << times << ":" << endl;
//			for (int i = 0; i < N; ++i) {
//				cout << "w[" << i << "] = " << w[i] << endl;
//			}
//			cout << "b = " << b << endl;
//			cout << endl;
//#endif // DEBUG

			// �жϵ���Ч��
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

				// �������Ž��
				for (int i = 0; i < N; ++i) {
					tmpW[i] = w[i];
				}
				tmpB = b;

			}
			else {

				// �����ü����Ҳ�����˵���Ѿ���������
				++idx;
				if (idx > evaluate) break;

			}

			if (idx < evaluate) {

				// �������w��b
				for (int i = 0; i < N; ++i) {
					w[i] = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;
				}
				b = ((rand() / static_cast<double>(RAND_MAX)) - 0.5) * myMax;

			}
			// ����������һ
			++times;

		}

		// �����Ž����������
		for (int i = 0; i < N; ++i) w[i] = tmpW[i];
		b = tmpB;

	}

	// ���Իع�Ԥ�⺯��
	double predict(vector<double> x) {
		
		// ������
		double ans = 0;

		for (int i = 0; i < N; ++i) {
			ans += w[i] * x[i];
		}

		return ans + b;

	}

	// ˽�г�Ա
private:

	// ���Իع�ģ�͵Ĳ���
	double w[N];
	double b;

};

