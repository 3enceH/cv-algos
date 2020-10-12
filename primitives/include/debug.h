#pragma once

#include <opencv2/core.hpp>
#include <random>

#include "core.h"

void EXPORT labelsToColored(const cv::Mat& labeled, cv::Mat& colored);

void EXPORT splitChannels(const cv::Mat& multi, cv::Mat& ch1, cv::Mat& ch2);
void EXPORT splitChannels(const cv::Mat& multi, cv::Mat& ch1, cv::Mat& ch2, cv::Mat& ch3);

void EXPORT joinChannels(const cv::Mat& ch1, const cv::Mat& ch2, const cv::Mat& ch3, cv::Mat& joined);
void EXPORT joinChannels(const cv::Mat& ch1, const cv::Mat& ch2, cv::Mat& joined);

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


