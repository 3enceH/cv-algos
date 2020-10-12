#include <numeric>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/highgui.hpp>

#include "ccl.h"
#include "debug.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

class CCLabelingAlgorithmicTest {
public:
    void operator()(int width, int height) {
        ConnectedComponentLabeling cclabeling;
        cv::Mat black(height, width, CV_32FC1, cv::Scalar(0));
        cv::Mat blackOut(height, width, CV_32SC1);
        cclabeling.applyOnImage(black, blackOut);

        EXPECT_TRUE(std::all_of(blackOut.begin<int>(), blackOut.end<int>(), [=](int val) { return val == 0; } ));

        std::default_random_engine gen;
        std::uniform_int_distribution<int> distX(0, width - 1);
        std::uniform_int_distribution<int> distY(0, height - 1);
        std::uniform_int_distribution<int> radius(std::min(width, height) / 20, std::min(width, height) / 4);

        cv::Mat blobs(height, width, CV_32FC1, cv::Scalar(0));
        cv::Mat blobsOut(height, width, CV_32SC1);

        for (int i = 1; i <= 10; i++) {
            int cx = distX(gen);
            int cy = distY(gen);
            int r = radius(gen);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if( r > (int)std::hypot(cx -x, cy - y)) {
                        ((float*)blobs.data)[y * width + x] = (float)10 * i;
                    }
                }
            }
        }

        cclabeling.applyOnImage(blobs, blobsOut);
        const float* const values = (float*)blobs.data;
        const int* const labels = (int*)blobsOut.data;

        // debug
        cv::Mat colored;
        labelsToColored(blobsOut, colored);
        cv::Mat r, g, b;
        splitChannels(colored, r, g, b);

        std::map<int, float> mappedValues;
        bool ok = true;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const float& value = values[OFFSET_ROW_MAJOR(x, y, width, 1)];
                const int& label = labels[OFFSET_ROW_MAJOR(x, y, width, 1)];
                if (mappedValues.find(labels[OFFSET_ROW_MAJOR(x, y, width, 1)]) == mappedValues.end()) {
                    mappedValues[label] = value;
                    continue;
                }
                if (mappedValues[label] != value) {
                    ok = false;
                    break;
                }
            }
        }
        EXPECT_EQ(ok, true);
    }
};

TEST(CCLabelingTest, Precomputed) {
    CCLabelingAlgorithmicTest test;
    test(4 * 4, 3 * 4);
    test(4 * 4 * 4, 3 * 4 * 4);
    test(800, 480);
}
