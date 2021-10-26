#include <numeric>

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "debug.h"
#include "gradient.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class GradientOpenCVComparisonTest
{
public:
    cv::Mat squares(int width, int height)
    {
        cv::Mat out(height, width, CV_8UC1);
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                uchar value = ((x / 32 + y / 32) % 2 == 0) ? 255 : 0;
                out.data[y * width + x] = value;
            }
        }
        return std::move(out);
    }

    cv::Mat gradient(int width, int height)
    {
        cv::Mat out(height, width, CV_8UC1);
        const int whalf = width / 2;
        const int hhalf = height / 2;
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                int xx = x >= whalf ? width - 1 - x : x;
                int yy = y >= hhalf ? height - 1 - y : y;
                uchar value = (uchar)(((xx % whalf) / (float)whalf) * 127 + ((yy % hhalf) / (float)hhalf) * 127);
                out.data[y * width + x] = value;
            }
        }
        return std::move(out);
    }

    void check(cv::Mat& mag1, cv::Mat& mag2, cv::Mat& theta1, cv::Mat& theta2)
    {
        {
            cv::Mat diffMag;
            cv::absdiff(mag1, mag2, diffMag);
            double diff = cv::sum(diffMag)[0];

            double avgDiffPerPix = diff / ((double)mag1.cols * mag1.rows);
            EXPECT_TRUE(avgDiffPerPix < 1.f);

            float maxDiffPerPix = 0;
            float epsilon = 255.f / 4;
            bool equals = std::all_of(diffMag.begin<float>(), diffMag.end<float>(), [&](float val) {
                if(val > maxDiffPerPix)
                {
                    maxDiffPerPix = val;
                }
                return val < epsilon;
            });
            EXPECT_TRUE(equals);
            std::cout << "avgDiffPerPix " << std::setw(10) << std::setprecision(5) << avgDiffPerPix << " maxDiffPerPix "
                      << std::setw(5) << maxDiffPerPix << std::endl;
        }
        {
            cv::Mat diffTheta;
            cv::absdiff(theta1, theta2, diffTheta);
            double diff = cv::sum(diffTheta)[0];

            double avgDiffPerPix = diff / ((double)theta1.cols * theta1.rows);
            EXPECT_TRUE(avgDiffPerPix < 1.f);

            float maxDiffPerPix = 0;
            float epsilon = 255.f / 4;
            bool equals = std::all_of(diffTheta.begin<float>(), diffTheta.end<float>(), [&](float val) {
                if(val > maxDiffPerPix)
                {
                    maxDiffPerPix = val;
                }
                return val < epsilon;
            });
            EXPECT_TRUE(equals);
            std::cout << "avgDiffPerPix " << std::setw(10) << std::setprecision(5) << avgDiffPerPix << " maxDiffPerPix "
                      << std::setw(5) << maxDiffPerPix << std::endl;
        }
    }

    void generated(cv::Mat input)
    {
        std::cout << "generated dims " << input.cols << "x" << input.rows << std::endl;

        cv::GaussianBlur(input, input, cv::Size(5, 5), 1.f);

        cv::Mat mag1, theta1, sobelX, sobelY;

        cv::Sobel(input, sobelX, CV_32FC1, 1, 0);
        cv::Sobel(input, sobelY, CV_32FC1, 0, 1);
        combine<float>(sobelX, sobelY, mag1, [](float a, float b) -> float { return (float)hypot(a, b); });
        combine<float>(sobelX, sobelY, theta1, [](float a, float b) -> float { return (float)atan2(b, a); });

        cv::Mat tmp, mag2, theta2;
        Gradient cpuFilter;
        cpuFilter.apply(input, tmp);

        splitChannels(tmp, mag2, theta2);

        // cv::Mat out1u8, out2u8;
        // cv::convertScaleAbs(mag, out2u8);
        // cv::convertScaleAbs(out1, out1u8);

        check(mag1, mag2, theta1, theta2);
    }

    void picture(const std::string& name)
    {

        std::string filename = std::string(STR(DATA_ROOT)) + "/" + name;
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);

        std::cout << "pic " << filename << " dims " << pic.cols << "x" << pic.rows << std::endl;

        cv::GaussianBlur(pic, pic, cv::Size(5, 5), 1.f);

        cv::Mat mag1, theta1, sobelX, sobelY;

        cv::Sobel(pic, sobelX, CV_32FC1, 1, 0);
        cv::Sobel(pic, sobelY, CV_32FC1, 0, 1);
        combine<float>(sobelX, sobelY, mag1, [](float a, float b) -> float { return (float)hypot(a, b); });
        combine<float>(sobelX, sobelY, theta1, [](float a, float b) -> float { return (float)atan2(b, a); });

        cv::Mat tmp, mag2, theta2;
        Gradient cpuFilter;
        cpuFilter.apply(pic, tmp);

        splitChannels(tmp, mag2, theta2);

        check(mag1, mag2, theta1, theta2);
    }
};

TEST(GradientTest, OpenCVComparison)
{
    GradientOpenCVComparisonTest test;
    test.generated(test.gradient(64, 64));
    test.generated(test.gradient(10 * 4 * 32, 10 * 4 * 32));
    test.generated(test.squares(64, 64));
    test.generated(test.squares(10 * 4 * 32, 10 * 4 * 32));
    test.picture("samuraijack.jpg");
    test.picture("rihanna.jpg");
}