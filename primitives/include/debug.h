#pragma once

#include <opencv2/core.hpp>
#include <random>

#include "core.h"

void EXPORT labelsToColored(const cv::Mat& labeled, cv::Mat& colored);

void EXPORT splitChannels(const cv::Mat& multi, cv::Mat& ch1, cv::Mat& ch2);
void EXPORT splitChannels(const cv::Mat& multi, cv::Mat& ch1, cv::Mat& ch2, cv::Mat& ch3);

void EXPORT joinChannels(const cv::Mat& ch1, const cv::Mat& ch2, const cv::Mat& ch3, cv::Mat& joined);
void EXPORT joinChannels(const cv::Mat& ch1, const cv::Mat& ch2, cv::Mat& joined);

template <typename T>
void combine(const cv::Mat& input1, cv::Mat& input2, cv::Mat& output, std::function<T(T value1, T value2)> perPixelOp) {
    int channels = input1.channels();
    int width = input1.cols;
    int height = input1.rows;
    const T* const data1 = (T*)input1.data;
    const T* const data2 = (T*)input2.data;

    output = cv::Mat(height, width, input1.type());
    T* const out = (T*)output.data;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                size_t idx = OFFSET_ROW_MAJOR(x, y, width, c + 1);
                out[idx] = perPixelOp(data1[idx], data2[idx]);
            }
        }
    }
}

//void combine(const cv::Mat& input, cv::Mat& output, std::function<uchar(uchar value1, uchar value2)> channelOp) {
//    int channels = input.channels();
//    std::vector<uchar> buffer(channels);
//
//    int width = input.cols;
//    int height = input.rows;
//    int baseType = input.type() - ((channels - 1) << CV_CN_SHIFT);
//    const uchar* const in = (uchar*)input.data;
//
//    output = cv::Mat(height, width, baseType);
//    uchar* const out = (uchar*)output.data;
//
//    auto asdf = [&](int x, int y) {
//        out[OFFSET_ROW_MAJOR(x, y, width, 1)] = channelOp(buffer[0], buffer[0]);
//    };
//
//    for (int y = 0; y < height; y++) {
//        for (int x = 0; x < width; x++) {
//            for (int c = 0; c < channels; c++) {
//                buffer[c] = in[OFFSET_ROW_MAJOR(x, y, width, channels) + c];
//            }   
//                
//            switch (channels) {
//            case 1: 
//                out[OFFSET_ROW_MAJOR(x, y, width, 1)] = channelOp(buffer[0]);
//                break;
//            case 2:
//                out[OFFSET_ROW_MAJOR(x, y, width, 1)] = channelOp(buffer[0], buffer[1]);
//                break;
//            case 3:
//                out[OFFSET_ROW_MAJOR(x, y, width, 1)] = channelOp(buffer[0], buffer[1], buffer[2]);
//                break;
//            case 4:
//                out[OFFSET_ROW_MAJOR(x, y, width, 1)] = channelOp(buffer[0], buffer[1], buffer[2], buffer[3]);
//                break;
//            }
//            
//        }
//    }
//} 


#include <map>
#include <chrono>

class PerformanceTimer {

private:
	std::map<std::chrono::steady_clock::time_point, std::string> log;

public:
    void EXPORT tag(const std::string& name);
	void clear() { log.clear(); }
    void EXPORT start();
    std::string EXPORT summary() const;
};


