#include <numeric>

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "sobelfilter.h"
#include "debug.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class GaussianFilterOpenCVComparisonTest {
public:
    void generated(int width, int height) {

        cv::Mat squares(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar value = ((x / (width / 4) + y / (height / 4)) % 2 == 0) ? 255 : 0;
                squares.data[y * width + x] = value;
            }
        }
        cv::GaussianBlur(squares, squares, cv::Size(5, 5), 1.f);

        cv::Mat out1, out2, outX, outY;

        cv::Sobel(squares, outX, CV_32FC1, 1, 0);
        cv::Sobel(squares, outY, CV_32FC1, 0, 1);

        addWeighted(outX, 0.5, outY, 0.5, 0, out1);

        SobelFilter cpuFilter;
        cpuFilter.applyOnImage(squares, out2);

        cv::Mat mag, grad;
        splitChannels(out2, mag, grad);


        //int epsilon = 3;
        //int diff = std::accumulate(out1.begin<uchar>(), out1.end<uchar>(), 0) - std::accumulate(out2.begin<uchar>(), out2.end<uchar>(), 0);
        //float diffPerPix = diff / ((double)width * height);
        //EXPECT_TRUE(diffPerPix < 1.f);
        //int maxDiff = 0;
        //bool equals = std::equal(out1.begin<uchar>(), out1.end<uchar>(), out2.begin<uchar>(), [&](uchar a, uchar b) {
        //    int diff = a > b ? a - b : b - a;
        //    if (diff > maxDiff) maxDiff = diff;
        //    return diff < epsilon;
        //    });
        //EXPECT_TRUE(equals);
    }

    void picture() {

        std::string filename = "c:\\Users\\Bence\\Downloads\\RbLtUqIP_o.jpg";
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);
        cv::Mat out1, out2;

        cv::GaussianBlur(pic, pic, cv::Size(5, 5), 1.f);

        SobelFilter cpuFilter;
        cpuFilter.applyOnImage(pic, out2);

        int epsilon = 3;
        int diff = std::accumulate(out1.begin<uchar>(), out1.end<uchar>(), 0) - std::accumulate(out2.begin<uchar>(), out2.end<uchar>(), 0);
        double diffPerPix = diff / ((double)pic.cols * pic.rows);
        EXPECT_TRUE(diffPerPix < 1.f);
        int maxDiff = 0;
        bool equals = std::equal(out1.begin<uchar>(), out1.end<uchar>(), out2.begin<uchar>(), [&](uchar a, uchar b) {
            int diff = a > b ? a - b : b - a;
            if (diff > maxDiff) maxDiff = diff;
            return diff < epsilon;
        });
        EXPECT_TRUE(equals);
    }
};

TEST(GaussianFilterTest, OpenCVComparison) {
    GaussianFilterOpenCVComparisonTest test;
    test.generated(32, 32);
    test.generated(10 * 4 * 32, 10 * 4 * 32);
    //test.picture();
    //test.picture(2, 1.f);
}