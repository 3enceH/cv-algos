#include "debug.h"

#include <sstream>
#include <iomanip>

void PerformanceTimer::tag(const std::string& name) {
    auto now = std::chrono::steady_clock::now();
    log[now] = name;
}

void PerformanceTimer::start() {
    clear();
    log[std::chrono::steady_clock::now()] = "start";
}

std::string PerformanceTimer::summary() const {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(log.crbegin()->first - log.cbegin()->first).count();

    int strLengthMax = 0;
    for (auto& it : log) {
        if (it.second.length() > strLengthMax) strLengthMax = (int)it.second.length();
    }

    std::ostringstream stream;

    auto cur = log.begin();
    cur++;
    auto last = log.begin();
    while (cur != log.cend()) {
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(cur->first - last->first).count();
        float ratio = 100.f * diff / (float)elapsed;
        stream << std::setw(strLengthMax) << std::left << cur->second << ": " << std::setw(8) << std::left << diff << "ms " << ratio << "%" << std::endl;
        cur++;
        last++;
    }
    stream << std::setw(strLengthMax) << std::left << "SUM" << ": " << std::setw(8) << std::left << elapsed << "ms " << 100 << "%" << std::endl;
    return std::move(stream.str());
}

void labelsToColored(const cv::Mat& labeled, cv::Mat& colored) {
    int* labels = (int*)labeled.data;

    if (colored.empty()) {
        colored = std::move(cv::Mat(labeled.rows, labeled.cols, CV_8UC3));
    }
    uchar* color = (uchar*)colored.data;

    std::default_random_engine gen;
    std::uniform_int_distribution<int> byte(0, 255);

    for (int y = 0; y < labeled.rows; y++) {
        for (int x = 0; x < labeled.cols; x++) {
            gen.seed(labels[OFFSET_ROW_MAJOR(x, y, labeled.cols, 1)]);
            color[OFFSET_ROW_MAJOR(x, y, labeled.cols, 3) + 0] = byte(gen);
            color[OFFSET_ROW_MAJOR(x, y, labeled.cols, 3) + 1] = byte(gen);
            color[OFFSET_ROW_MAJOR(x, y, labeled.cols, 3) + 2] = byte(gen);
        }
    }
}

void splitChannels(const cv::Mat* const multi, cv::Mat* ch1, cv::Mat* ch2, cv::Mat* ch3) {
    //+-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +
    //|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
    //+-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +
    //| CV_8U | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 |
    //| CV_8S | 1 | 9 | 17 | 25 | 33 | 41 | 49 | 57 |
    //| CV_16U | 2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
    //| CV_16S | 3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
    //| CV_32S | 4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
    //| CV_32F | 5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
    //| CV_64F | 6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
    //+-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +

    assert(multi != nullptr);
    assert(ch1 != nullptr);
    assert(ch2 != nullptr);

    int channels = multi->channels();
    int baseType = multi->type() - ((channels - 1) << CV_CN_SHIFT);
    int width = multi->cols;
    int height = multi->rows;
    size_t elemSize = multi->elemSize1();
    std::vector<uchar> bytes(elemSize);

    *ch1 = cv::Mat(height, width, baseType);
    *ch2 = cv::Mat(height, width, baseType);
    if (ch3 != nullptr)
        *ch3 = cv::Mat(height, width, baseType);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t multiIdx = OFFSET_ROW_MAJOR(x, y, width, channels) * elemSize;
            size_t singleIdx = OFFSET_ROW_MAJOR(x, y, width, 1) * elemSize;

            memcpy(bytes.data(), &multi->data[multiIdx + 0 * elemSize], bytes.size());
            memcpy(&ch1->data[singleIdx], bytes.data(), bytes.size());

            memcpy(bytes.data(), &multi->data[multiIdx + 1 * elemSize], bytes.size());
            memcpy(&ch2->data[singleIdx], bytes.data(), bytes.size());

            if (ch3 != nullptr) {
                memcpy(bytes.data(), &multi->data[multiIdx + 2 * elemSize], bytes.size());
                memcpy(&ch3->data[singleIdx], bytes.data(), bytes.size());
            }
        }
    }
}


void splitChannels(const cv::Mat& multi, cv::Mat& ch1, cv::Mat& ch2) {
    splitChannels(&multi, &ch1, &ch2, nullptr);
}

void splitChannels(const cv::Mat& multi, cv::Mat& ch1, cv::Mat& ch2, cv::Mat& ch3) {
    splitChannels(&multi, &ch1, &ch2, &ch3);
}

void joinChannels(cv::Mat* joined, const cv::Mat* ch1, const cv::Mat* ch2, const cv::Mat* ch3)
{
    assert(joined != nullptr);
    assert(ch1 != nullptr);
    assert(ch2 != nullptr);

    int channels = ch3 == nullptr ? 2 : 3;
    int joinedType = ch1->type() + ((channels - 1) << CV_CN_SHIFT);
    int width = ch1->cols;
    int height = ch1->rows;
    size_t elemSize = ch1->elemSize1();
    std::vector<uchar> bytes(elemSize);

    *joined = cv::Mat(height, width, joinedType);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t multiIdx = OFFSET_ROW_MAJOR(x, y, width, channels) * elemSize;
            size_t singleIdx = OFFSET_ROW_MAJOR(x, y, width, 1) * elemSize;
            
            memcpy(bytes.data(), &ch1->data[singleIdx], bytes.size());
            memcpy(&joined->data[multiIdx + 0 * elemSize], bytes.data(), bytes.size());

            memcpy(bytes.data(), &ch2->data[singleIdx], bytes.size());
            memcpy(&joined->data[multiIdx + 1 * elemSize], bytes.data(), bytes.size());

            if (ch3 != nullptr) {
                memcpy(bytes.data(), &ch3->data[singleIdx], bytes.size());
                memcpy(&joined->data[multiIdx + 2 * elemSize], bytes.data(), bytes.size());
            }
        }
    }
}


void joinChannels(const cv::Mat& ch1, const cv::Mat& ch2, const cv::Mat& ch3, cv::Mat& joined) {
    joinChannels(&joined, &ch1, &ch2, &ch3);
}

void joinChannels(const cv::Mat& ch1, const cv::Mat& ch2, cv::Mat& joined) {
    joinChannels(&joined, &ch1, &ch2, nullptr);
}
