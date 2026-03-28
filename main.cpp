#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

const int N = 6;            // 아미노산의 개수 (HPHHPH 서열 길이에 맞춤)
const int L = 6;            // 격자축의 크기
const int M = L * L;        // 총 격자점 수
const int K = N * M;        // 총 큐비트 수 (N * M)
const double LAMBDA = 15.0; // [조정됨] 고전 SA 탐색을 위한 최적 페널티 상수
const double DELTA = 2.0;   // 소수성 결합 보상

string sequence = "HPHHPH"; // 아미노산 서열

// 2차원 인덱스를 1차원으로
inline int get_idx(int i, int alpha) { return i * M + alpha; }

// 인접 여부 확인 (맨하튼 거리 = 1)
bool is_adjacent(int a, int b)
{
    int r1 = a / L, c1 = a % L;
    int r2 = b / L, c2 = b % L;
    return abs(r1 - r2) + abs(c1 - c2) == 1;
}

int main()
{
    vector<vector<double>> W(K, vector<double>(K, 0.0));

    // W matrix
    // H_1: 하나의 아미노산은 딱 한 곳에만 있어야 한다
    for (int i = 0; i < N; ++i)
    {
        for (int a = 0; a < M; ++a)
        {
            int u = get_idx(i, a);
            W[u][u] += -LAMBDA; // 1차항
            for (int b = a + 1; b < M; ++b)
            {
                int v = get_idx(i, b);
                W[u][v] += 2.0 * LAMBDA; // 2차항
            }
        }
    }

    // H_2: 하나의 격자점에는 두 개 이상 있으면 안된다
    for (int a = 0; a < M; ++a)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = i + 1; j < N; ++j)
            {
                W[get_idx(i, a)][get_idx(j, a)] += LAMBDA;
            }
        }
    }

    // H_3: 서열 상 이웃하면 격자 상에서 이웃해야한다
    for (int i = 0; i < N - 1; ++i)
    {
        for (int a = 0; a < M; ++a)
        {
            for (int b = 0; b < M; ++b)
            {
                if (a == b)
                    continue;
                if (!is_adjacent(a, b))
                {
                    int u = get_idx(i, a), v = get_idx(i + 1, b);
                    if (u < v)
                        W[u][v] += LAMBDA;
                    else
                        W[v][u] += LAMBDA;
                }
            }
        }
    }

    // H_4: H-H 붙으면 에너지 낮아지는거
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 2; j < N; ++j)
        {
            if (sequence[i] == 'H' && sequence[j] == 'H')
            {
                for (int a = 0; a < M; ++a)
                {
                    for (int b = 0; b < M; ++b)
                    {
                        if (is_adjacent(a, b))
                        {
                            int u = get_idx(i, a), v = get_idx(j, b);
                            if (u < v)
                                W[u][v] -= DELTA;
                            else
                                W[v][u] -= DELTA;
                        }
                    }
                }
            }
        }
    }

    // 가짜 어닐링
    vector<int> x(K, 0);
    random_device rd;
    mt19937 gen(rd()); // RNG
    uniform_real_distribution<> dis(0.0, 1.0);
    uniform_int_distribution<> bit_dis(0, K - 1);

    double T = 100.0;      // 초기 온도
    double T_min = 0.01;   // 최종 온도
    double alpha = 0.9999; // 냉각 속도
    int steps = 1500000;   // 탐색 횟수

    for (int s = 0; s < steps && T > T_min; ++s)
    {
        int k = bit_dis(gen); // 무작위 비트

        double delta_E = 0;
        int x_k_old = x[k];
        int x_k_new = 1 - x_k_old;
        int diff = x_k_new - x_k_old;

        // 대각 성분 1차항 기여
        delta_E += W[k][k] * diff;
        // 비대각 성분 2차항 기여
        for (int j = 0; j < K; ++j)
        {
            if (k == j)
                continue;
            double weight = (k < j) ? W[k][j] : W[j][k];
            delta_E += weight * diff * x[j];
        }

        // Metropolis 수용 조건
        if (delta_E < 0 || dis(gen) < exp(-delta_E / T))
        {
            x[k] = x_k_new;
        }
        T *= alpha; // 냉각
    }

    // 유효성 검증
    vector<int> pos(N, -1);
    bool one_hot_ok = true, overlap_ok = true, connectivity_ok = true;

    for (int i = 0; i < N; ++i)
    {
        int count = 0;
        for (int a = 0; a < M; ++a)
        {
            if (x[get_idx(i, a)] == 1)
            {
                pos[i] = a;
                count++;
            }
        }
        if (count != 1)
            one_hot_ok = false;
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            if (pos[i] != -1 && pos[i] == pos[j])
                overlap_ok = false;
        }
    }

    for (int i = 0; i < N - 1; ++i)
    {
        if (pos[i] == -1 || pos[i + 1] == -1 || !is_adjacent(pos[i], pos[i + 1]))
            connectivity_ok = false;
    }

    // --- 3. 결과 출력 ---
    cout << "\n====================================\n";
    if (one_hot_ok && overlap_ok && connectivity_ok)
    {
        cout << "✅ 검증: 물리적으로 유효함" << "\n";
    }
    else
    {
        cout << "❌ 검증: 물리적으로 유효하지 않음" << "\n";
        cout << "   (사유: " << (!one_hot_ok ? "위치오류 " : "")
             << (!overlap_ok ? "좌표중첩 " : "")
             << (!connectivity_ok ? "사슬끊어짐" : "") << ")" << "\n";
    }
    cout << "====================================\n";

    vector<string> grid(L, string(L, '.'));
    cout << "도출된 고유구조" << "\n";
    for (int i = 0; i < N; ++i)
    {
        if (pos[i] != -1)
        {
            grid[pos[i] / L][pos[i] % L] = sequence[i];
            cout << "아미노산 " << i + 1 << "(" << sequence[i] << ") ~ (" << pos[i] / L << "," << pos[i] % L << ")" << "\n";
        }
    }

    cout << "\n[2차원 격자]\n";
    for (int r = 0; r < L; ++r)
    {
        for (int c = 0; c < L; ++c)
            cout << grid[r][c] << " ";
        cout << "\n";
    }

    cout << "\n최종 온도: " << T << "\n";

    return 0;
}
