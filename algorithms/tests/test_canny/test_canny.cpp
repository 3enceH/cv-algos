#include <numeric>

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "canny.h"
#include "debug.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void check(cv::Mat& out1, cv::Mat& out2) {
    cv::Mat diffImg;
    cv::absdiff(out1, out2, diffImg);

    double avgDiffPerPix = cv::sum(diffImg)[0] / ((double)out1.cols * out1.rows);
    EXPECT_TRUE(avgDiffPerPix < 1.1f);

    int maxDiff = 0;
    int epsilon = 3;
    bool equals = std::all_of(diffImg.begin<uchar>(), diffImg.end<uchar>(), [&](uchar val) {
        if (val > maxDiff) {
            maxDiff = val;
        }
        return val < epsilon;
        });
    EXPECT_TRUE(equals);
    std::cout << "avgDiffPerPix " << std::setw(10) << std::setprecision(5) << avgDiffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
}

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
    const int hhalf = height / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int xx = x >= whalf ? width - 1 - x : x;
            int yy = y >= hhalf ? height - 1 - y : y;
            uchar value = (uchar)(((xx % whalf) / (float)whalf) * 127 + ((yy % hhalf) / (float)hhalf) * 127);
            out.data[y * width + x] = value;
        }
    }
    return std::move(out);
}

class GradientOpenCVComparisonTest {
public:
    void generated(cv::Mat& input) {

        cv::Mat out1, out2;

        float threshold1 = 70.f;
        float threshold2 = 140.f;
        cv::Canny(input, out1, threshold1, threshold2);

        Canny canny(threshold1, threshold2);
        canny.apply(input, out2);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);

        check(out1, out2);
    }

    void picture(const std::string& name) {

        std::string filename = std::string(STR(DATA_ROOT)) + "/" + name;
        cv::Mat pic = cv::imread(filename), grey;
        cv::cvtColor(pic, grey, cv::COLOR_RGB2GRAY);

        cv::Mat out1, out2;

        float low = 70.f;
        float high = 140.f;
        cv::Canny(grey, out1, low, high);

        Canny canny(low, high);
        canny.apply(grey, out2);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);

        cv::Mat lines;
        joinChannels(grey, out2, lines);

        check(out1, out2);
    }
};

TEST(GradientTest, OpenCVComparison) {
    GradientOpenCVComparisonTest test;
    test.generated(squares(64, 64));
    test.generated(squares(10 * 4 * 32, 10 * 4 * 32));
    test.generated(gradient(64, 64));
    test.generated(gradient(10 * 4 * 32, 10 * 4 * 32));
    test.picture("samuraijack.jpg");
    test.picture("rihanna.jpg");
}