#include <numeric>

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "gradient.h"
#include "debug.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class GradientOpenCVComparisonTest {
public:

    cv::Mat squares(int width, int height) {
        cv::Mat out(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar value = ((x / 32 + y / 32) % 2 == 0) ? 255 : 0;
                out.data[y * width + x] = value;
            }
        }
        return std::move(out);
    }

    cv::Mat gradient(int width, int height) {
        cv::Mat out(height, width, CV_8UC1);
        const int whalf = width / 2;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int xx = x >= whalf ? width - 1 - x : x;
                uchar value = (uchar)(((xx % whalf) / (float)whalf) * 255);
                out.data[y * width + x] = value;
            }
        }
        return std::move(out);
    }

    void generated(cv::Mat input) {
        cv::GaussianBlur(input, input, cv::Size(5, 5), 1.f);

        cv::Mat out1, out2, outX, outY;

        cv::Sobel(input, outX, CV_32FC1, 1, 0);
        cv::Sobel(input, outY, CV_32FC1, 0, 1);
        combine<float>(outX, outY, out1, [](float a, float b) -> float { return (float)hypot(a, b); });

        Gradient cpuFilter;
        cpuFilter.apply(input, out2);

        cv::Mat mag, grad;
        splitChannels(out2, mag, grad);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);
        
        cv::Mat diffImg;
        cv::absdiff(out1, mag, diffImg);
        double diff = cv::sum(diffImg)[0];

        double avgDiffPerPix = diff / ((double)input.cols * input.rows);
        EXPECT_TRUE(avgDiffPerPix < 1.f);

        float maxDiffPerPix = 0;
        float epsilon = 255.f / 4;
        bool equals = std::all_of(diffImg.begin<float>(), diffImg.end<float>(), [&](float val) {
            if (val > maxDiffPerPix) {
                maxDiffPerPix = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "avgDiffPerPix " << std::setw(10) << std::setprecision(5) << avgDiffPerPix << " maxDiffPerPix " << std::setw(5) << maxDiffPerPix << std::endl;
    }

    void picture(const std::string& name) {

        std::string filename = std::string(STR(DATA_ROOT)) + "/" + name;
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);

        cv::GaussianBlur(pic, pic, cv::Size(5, 5), 1.f);

        cv::Mat out1, out2, outX, outY;

        cv::Sobel(pic, outX, CV_32FC1, 1, 0);
        cv::Sobel(pic, outY, CV_32FC1, 0, 1);
        combine<float>(outX, outY, out1, [](float a, float b) -> float { return (float)hypot(a, b); });

        Gradient cpuFilter;
        cpuFilter.apply(pic, out2);

        cv::Mat mag, grad;
        splitChannels(out2, mag, grad);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);

        int width = pic.cols;
        int height = pic.rows;

        cv::Mat diffImg;
        cv::absdiff(out1, mag, diffImg);
        double diff = cv::sum(diffImg)[0];

        double avgDiffPerPix = diff / ((double)width * height);
        EXPECT_TRUE(avgDiffPerPix < 1.f);

        float maxDiffPerPix = 0;
        float epsilon = 255.f/4;
        bool equals = std::all_of(diffImg.begin<float>(), diffImg.end<float>(), [&](float val) {
            if (val > maxDiffPerPix) {
                maxDiffPerPix = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "avgDiffPerPix " << std::setw(10) << std::setprecision(5) << avgDiffPerPix << " maxDiffPerPix " << std::setw(5) << maxDiffPerPix << std::endl;
    }
};

TEST(GradientTest, OpenCVComparison) {
    GradientOpenCVComparisonTest test;
    test.generated(test.gradient(64, 64));
    test.generated(test.gradient(10 * 4 * 32, 10 * 4 * 32));
    test.generated(test.squares(64, 64));
    test.generated(test.squares(10 * 4 * 32, 10 * 4 * 32));
    test.picture("samuraijack.jpg");
    test.picture("rihanna.jpg");
}