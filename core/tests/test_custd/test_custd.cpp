#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>


inline cudaError_t cpu_malloc(void** devPtr, size_t size)
{
    *devPtr = malloc(size);
    return cudaSuccess;
}
inline cudaError_t cpu_free(void* devPtr)
{
    free(devPtr);
    return cudaSuccess;
}

inline cudaError_t cpu_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind)
{
    (void)kind;
    memcpy(dst, src, size);
    return cudaSuccess;
}

inline cudaError_t cpu_memset(void* devPtr, int value, size_t size)
{
    memset(devPtr, value, size);
    return cudaSuccess;
}

#define cudaMalloc cpu_malloc
#define cudaFree   cpu_free
#define cudaMemcpy cpu_memcpy
#define cudaMemset cpu_memset

#include "custd.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

struct Rect
{
    float x, y;

    Rect(float x, float y) : x(x), y(y) {}
    Rect(int x, int y) : x((float)x), y((float)y) {}

    bool operator<(const Rect& other) { return hypotf(x, y) < hypotf(other.x, other.y); }
    bool operator==(const Rect& other) { return hypotf(x, y) == hypotf(other.x, other.y); }
    bool operator<=(const Rect& other) { return *this < other || *this == other; }
    bool operator>(const Rect& other) { return !(*this <= other); }
};

std::ostream& operator<<(std::ostream& stream, const Rect& rect)
{
    stream << "Rect[" << rect.x << "," << rect.y << "]";
    return stream;
}

class CuStdVectorTest
{
public:
    void operator()()
    {
        {

            // simple arithmetic type vector
            custd::vector<int, 10> v;

            {
                // push, pop one element, clear
                v.push_back(1);
                EXPECT_EQ(v.size(), 1);
                EXPECT_EQ(v.end() - v.begin(), 1);
                v.push_back(2);
                EXPECT_EQ(v.size(), 2);
                EXPECT_EQ(v.end() - v.begin(), 2);
                v.push_back(3);
                EXPECT_EQ(v.size(), 3);
                EXPECT_EQ(v.end() - v.begin(), 3);
                v.push_back(4);
                EXPECT_EQ(v.size(), 4);
                EXPECT_EQ(v.end() - v.begin(), 4);
                v.pop_back();
                EXPECT_EQ(v.size(), 3);
                EXPECT_EQ(v.end() - v.begin(), 3);

                v.clear();
                EXPECT_EQ(v.size(), 0);
                EXPECT_EQ(v.end() - v.begin(), 0);
                EXPECT_EQ(v.empty(), true);
            }
            {
                // erase till end
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(i);
                }
                std::cout << v << std::endl;
                auto it = v.erase(v.begin() + 2, v.end());
                std::cout << v << std::endl;
                EXPECT_EQ(it, v.begin() + 2);
                EXPECT_EQ(v.size(), 2);
                EXPECT_EQ(*(v.begin()), 0);
                EXPECT_EQ(*(v.end() - 1), 1);
            }
            {
                // erase inner elements
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(i);
                }
                std::cout << v << std::endl;
                auto it = v.erase(v.begin() + 2, v.end() - 2);
                std::cout << v << std::endl;
                EXPECT_EQ(it, v.begin() + 2);
                EXPECT_EQ(v.size(), 4);
                EXPECT_EQ(*(v.begin()), 0);
                EXPECT_EQ(*(v.end() - 1), 9);
                EXPECT_EQ(*(v.begin() + 2), 8);
                EXPECT_EQ(*(v.end() - 3), 1);
            }
            {
                // erase from beginning
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(i);
                }
                std::cout << v << std::endl;
                auto it = v.erase(v.begin(), v.end() - 2);
                std::cout << v << std::endl;
                EXPECT_EQ(it, v.begin());
                EXPECT_EQ(v.size(), 2);
                EXPECT_EQ(*(v.begin()), 8);
                EXPECT_EQ(*(v.end() - 1), 9);
            }
            {
                // erase one element
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(i);
                }
                std::cout << v << std::endl;
                v.erase(v.begin() + 5);
                std::cout << v << std::endl;
                EXPECT_EQ(v.size(), 9);
                EXPECT_EQ(*(v.begin()), 0);
                EXPECT_EQ(*(v.end() - 1), 9);
                EXPECT_EQ(*(v.begin() + 5), 6);
            }

            {
                // sorting fixed random
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(rand() % 100);
                }
                EXPECT_EQ(v.size(), v.capacity());
                EXPECT_EQ(v.end() - v.begin(), v.capacity());

                custd::sort(v.begin(), v.end());
                for(int i = 1; i < v.size(); i++)
                {
                    EXPECT_TRUE(v[i - 1] <= v[i]);
                }
            }

            {
                // sorting pseudo random
                v.clear();
                srand((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(rand() % 100);
                }
                EXPECT_EQ(v.size(), v.capacity());
                EXPECT_EQ(v.end() - v.begin(), v.capacity());

                custd::sort(v.begin(), v.end());
                for(int i = 1; i < v.size(); i++)
                {
                    EXPECT_TRUE(v[i - 1] <= v[i]);
                }
            }
        }
        {
            // custom type vector sorting
            custd::vector<Rect, 10> v;
            {
                // push, pop one element, clear
                v.push_back(Rect(1, 1));
                EXPECT_EQ(v.size(), 1);
                EXPECT_EQ(v.end() - v.begin(), 1);
                v.push_back(Rect(2, 2));
                EXPECT_EQ(v.size(), 2);
                EXPECT_EQ(v.end() - v.begin(), 2);
                v.push_back(Rect(3, 3));
                EXPECT_EQ(v.size(), 3);
                EXPECT_EQ(v.end() - v.begin(), 3);
                v.push_back(Rect(4, 4));
                EXPECT_EQ(v.size(), 4);
                EXPECT_EQ(v.end() - v.begin(), 4);
                v.pop_back();
                EXPECT_EQ(v.size(), 3);
                EXPECT_EQ(v.end() - v.begin(), 3);

                v.clear();
                EXPECT_EQ(v.size(), 0);
                EXPECT_EQ(v.end() - v.begin(), 0);
                EXPECT_EQ(v.empty(), true);
            }
            {
                // erase till end
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(Rect(i, i));
                }
                std::cout << v << std::endl;
                auto it = v.erase(v.begin() + 2, v.end());
                std::cout << v << std::endl;
                EXPECT_EQ(it, v.begin() + 2);
                EXPECT_EQ(v.size(), 2);
                EXPECT_TRUE(*(v.begin()) == Rect(0, 0));
                EXPECT_TRUE(*(v.end() - 1) == Rect(1, 1));
            }
            {
                // erase inner elements
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(Rect(i, i));
                }
                std::cout << v << std::endl;
                auto it = v.erase(v.begin() + 2, v.end() - 2);
                std::cout << v << std::endl;
                EXPECT_EQ(it, v.begin() + 2);
                EXPECT_EQ(v.size(), 4);
                EXPECT_TRUE(*(v.begin()) == Rect(0, 0));
                EXPECT_TRUE(*(v.end() - 1) == Rect(9, 9));
                EXPECT_TRUE(*(v.begin() + 2) == Rect(8, 8));
                EXPECT_TRUE(*(v.end() - 3) == Rect(1, 1));
            }
            {
                // erase from beginning
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(Rect(i, i));
                }
                std::cout << v << std::endl;
                auto it = v.erase(v.begin(), v.end() - 2);
                std::cout << v << std::endl;
                EXPECT_EQ(it, v.begin());
                EXPECT_EQ(v.size(), 2);
                EXPECT_TRUE(*(v.begin()) == Rect(8, 8));
                EXPECT_TRUE(*(v.end() - 1) == Rect(9, 9));
            }
            {
                // erase one element
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back(Rect(i, i));
                }
                std::cout << v << std::endl;
                v.erase(v.begin() + 5);
                std::cout << v << std::endl;
                EXPECT_EQ(v.size(), 9);
                EXPECT_TRUE(*(v.begin()) == Rect(0, 0));
                EXPECT_TRUE(*(v.end() - 1) == Rect(9, 9));
                EXPECT_TRUE(*(v.begin() + 5) == Rect(6, 6));
            }

            {
                v.clear();
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back({(float)(rand() % 100), (float)(rand() % 100)});
                }
                EXPECT_EQ(v.size(), v.capacity());
                EXPECT_EQ(v.end() - v.begin(), v.capacity());

                custd::sort(v.begin(), v.end());
                for(int i = 1; i < v.size(); i++)
                {
                    EXPECT_TRUE(v[i - 1] <= v[i]);
                }
            }

            {
                v.clear();
                srand((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
                for(int i = 0; i < v.capacity(); i++)
                {
                    v.push_back({(float)(rand() % 100), (float)(rand() % 100)});
                }
                EXPECT_EQ(v.size(), v.capacity());
                EXPECT_EQ(v.end() - v.begin(), v.capacity());

                custd::sort(v.begin(), v.end());
                for(int i = 1; i < v.size(); i++)
                {
                    EXPECT_TRUE(v[i - 1] <= v[i]);
                }
            }
        }
    }
};

class CuStdMapTest
{
public:
    void operator()()
    {
        {
            // simple arithmetic value type
            custd::map<int, float, 10> m;

            m[0] = 0;
            EXPECT_EQ(m.size(), 1);
            EXPECT_EQ(m.end() - m.begin(), 1);
            m[1] = 1;
            EXPECT_EQ(m.size(), 2);
            EXPECT_EQ(m.end() - m.begin(), 2);
            m[2] = 2;
            EXPECT_EQ(m.size(), 3);
            EXPECT_EQ(m.end() - m.begin(), 3);
            auto count = m.erase(1);
            EXPECT_EQ(count, 1);
            EXPECT_EQ(m.size(), 2);
            EXPECT_EQ(m.end() - m.begin(), 2);

            m.clear();
            EXPECT_EQ(m.size(), 0);
            EXPECT_EQ(m.end() - m.begin(), 0);
            EXPECT_EQ(m.empty(), true);

            {
                // erase till end
                m.clear();
                for(int i = 0; i < m.capacity(); i++)
                {
                    m[i] = (float)i;
                }
                std::cout << m << std::endl;
                auto it = m.erase(m.begin() + 2, m.end());
                std::cout << m << std::endl;
                EXPECT_EQ(it, m.begin() + 2);
                EXPECT_EQ(m.size(), 2);
                EXPECT_EQ((m.begin())->first, 0);
                EXPECT_EQ((m.end() - 1)->first, 1);
            }
            {
                // erase inner elements
                m.clear();
                for(int i = 0; i < m.capacity(); i++)
                {
                    m[i] = (float)i;
                }
                std::cout << m << std::endl;
                auto it = m.erase(m.begin() + 2, m.end() - 2);
                std::cout << m << std::endl;
                EXPECT_EQ(it, m.begin() + 2);
                EXPECT_EQ(m.size(), 4);
                EXPECT_EQ((m.begin())->first, 0);
                EXPECT_EQ((m.end() - 1)->first, 9);
                EXPECT_EQ((m.begin() + 2)->first, 8);
                EXPECT_EQ((m.end() - 3)->first, 1);
            }
            {
                // erase from beginning
                m.clear();
                for(int i = 0; i < m.capacity(); i++)
                {
                    m[i] = (float)i;
                }
                std::cout << m << std::endl;
                auto it = m.erase(m.begin(), m.end() - 2);
                std::cout << m << std::endl;
                EXPECT_EQ(it, m.begin());
                EXPECT_EQ(m.size(), 2);
                EXPECT_EQ((m.begin())->first, 8);
                EXPECT_EQ((m.end() - 1)->first, 9);
            }
            {
                // erase one element
                m.clear();
                for(int i = 0; i < m.capacity(); i++)
                {
                    m[i] = (float)i;
                }
                std::cout << m << std::endl;
                m.erase(m.begin() + 5);
                std::cout << m << std::endl;
                EXPECT_EQ(m.size(), 9);
                EXPECT_EQ((m.begin())->first, 0);
                EXPECT_EQ((m.end() - 1)->first, 9);
                EXPECT_EQ((m.begin() + 5)->first, 6);
            }
            {
                m.clear();
                for(int i = 0; i < m.capacity(); i++)
                {
                    m[(rand() % 1000)] = (rand() % 100) / 3.14f;
                }
                EXPECT_EQ(m.size(), m.capacity());
                EXPECT_EQ(m.end() - m.begin(), m.capacity());

                std::cout << m << std::endl;
                for(auto it = m.begin() + 1; it != m.end(); ++it)
                {
                    EXPECT_TRUE(*(it - 1) <= *it);
                }
            }

            {
                m.clear();
                srand((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
                for(int i = 0; i < m.capacity(); i++)
                {
                    m[(rand() % 1000)] = (rand() % 100) / 3.14f;
                }
                EXPECT_EQ(m.size(), m.capacity());
                EXPECT_EQ(m.end() - m.begin(), m.capacity());

                std::cout << m << std::endl;
                for(auto it = m.begin() + 1; it != m.end(); ++it)
                {
                    EXPECT_TRUE(*(it - 1) <= *it);
                }
            }
        }

        {
            // container value type
            custd::map<int, custd::vector<int, 10>, 10> m;
        }
        //{
        //    custd::map<int, custd::vector<int, 10>, 10> m;

        //    m[0].push_back(0);
        //    EXPECT_EQ(m.size(), 1);
        //    EXPECT_EQ(m.end() - m.begin(), 1);
        //    m[1].push_back(0);
        //    m[1].push_back(1);
        //    EXPECT_EQ(m.size(), 2);
        //    EXPECT_EQ(m.end() - m.begin(), 2);
        //    m[2].push_back(0);
        //    m[2].push_back(1);
        //    m[2].push_back(2);
        //    EXPECT_EQ(m.size(), 3);
        //    EXPECT_EQ(m.end() - m.begin(), 3);
        //    std::cout << m << std::endl;
        //    auto count = m.erase(1);
        //    std::cout << m << std::endl;
        //    EXPECT_EQ(count, 1);
        //    EXPECT_EQ(m.size(), 2);
        //    EXPECT_EQ(m.end() - m.begin(), 2);
        //}
        //{
        //    custd::map<int, custd::vector<int, 10>, 10> m;
        //    m.clear();
        //    for (int i = 0; i < m.capacity() / 2; i++) {
        //        int key = i;
        //        for (int j = 0; j < rand() % 5; j++) {
        //            m[key].push_back((rand() % 100) / 3.14f);
        //            std::cout << m << std::endl;
        //        }
        //    }
        //    EXPECT_EQ(m.size(), m.capacity() / 2);
        //    EXPECT_EQ(m.end() - m.begin(), m.capacity() / 2);

        //    std::cout << m << std::endl;
        //    for (auto it = m.begin() + 1; it != m.end(); ++it) {
        //        EXPECT_TRUE(*(it - 1) <= *it);
        //    }
        //}
    }
};

TEST(CuStd, vector)
{
    CuStdVectorTest test;
    test();
}

TEST(CuStd, map)
{
    CuStdMapTest test;
    test();
}
