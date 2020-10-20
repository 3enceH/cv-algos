#include <numeric>

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cannyedgedetector.h"
#include "debug.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class GradientOpenCVComparisonTest {
public:
    void generated(int width, int height) {

        cv::Mat squares(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                uchar value = ((x / (width / 4) + y / (height / 4)) % 2 == 0) ? 255 : 0;
                squares.data[y * width + x] = value;
            }
        }
        cv::Mat out1, out2;

        float threshold1 = 70.f;
        float threshold2 = 140.f;
        cv::Canny(squares, out1, threshold1, threshold2);

        CannyEdgeDetector canny(threshold1, threshold2);
        canny.apply(squares, out2);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);
        
        cv::Mat diffImg;
        cv::absdiff(out1, out2, diffImg);
        double diff = cv::sum(diffImg)[0];

        double diffPerPix = diff / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 1.f);

        float maxDiff = 0;
        float epsilon = 1.f;
        bool equals = std::all_of(diffImg.begin<float>(), diffImg.end<float>(), [&](float val) {
            if (val > maxDiff) {
                maxDiff = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "diffPerPix " << std::setw(10) << std::setprecision(5) << diffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
    }

    void picture() {

        std::string filename = "c:\\Users\\Bence\\Downloads\\RbLtUqIP_o.jpg";
        cv::Mat pic;
        cv::cvtColor(cv::imread(filename), pic, cv::COLOR_RGB2GRAY);

        cv::Mat out1, out2;

        float threshold1 = 70.f;
        float threshold2 = 140.f;
        cv::Canny(pic, out1, threshold1, threshold2);

        CannyEdgeDetector canny(threshold1, threshold2);
        canny.apply(pic, out2);

        //cv::Mat out1u8, out2u8;
        //cv::convertScaleAbs(mag, out2u8);
        //cv::convertScaleAbs(out1, out1u8);

        int width = pic.cols;
        int height = pic.rows;

        cv::Mat diffImg;
        cv::absdiff(out1, out2, diffImg);
        double diff = cv::sum(diffImg)[0];

        double diffPerPix = diff / ((double)width * height);
        EXPECT_TRUE(diffPerPix < 1.f);

        float maxDiff = 0;
        float epsilon = 52.f;
        bool equals = std::all_of(diffImg.begin<float>(), diffImg.end<float>(), [&](float val) {
            if (val > maxDiff) {
                maxDiff = val;
            }
            return val < epsilon;
            });
        EXPECT_TRUE(equals);
        std::cout << "diffPerPix " << std::setw(10) << std::setprecision(5) << diffPerPix << " maxDiff " << std::setw(5) << maxDiff << std::endl;
    }
};

TEST(GradientTest, OpenCVComparison) {
    GradientOpenCVComparisonTest test;
    test.generated(32, 32);
    //test.generated(10 * 4 * 32, 10 * 4 * 32);
    test.picture();
}