#include "ccl.h"

#include <map>

#define OFFSET(x, y) OFFSET_ROW_MAJOR(x, y, width, 1)

void mergeLabels(std::vector<int>& into, const std::vector<int>& what)
{
    std::vector<int> newElements;
    for(int e2 : what)
    {
        if(std::all_of(into.begin(), into.end(), [&](int e) { return e != e2; }))
        {
            newElements.emplace_back(e2);
        }
    }
    std::for_each(newElements.begin(), newElements.end(), [&](int e) { into.emplace_back(e); });
    std::sort(into.begin(), into.end());
}

void labelPixel(int x, int y, int width, int height, const float* values, int* labels, int& labelMax,
                std::map<int, std::vector<int>>& equivalentLabels)
{
    int currentLabel = labels[OFFSET(x, y)];
    int currentIdx = OFFSET(x, y);
    int westIdx = OFFSET(x - 1, y);
    int northIdx = OFFSET(x, y - 1);
    const float& current = values[currentIdx];

    float west = x - 1 >= 0 ? values[westIdx] : NAN;
    if(current == west)
    {
        labels[currentIdx] = labels[westIdx];
    }
    float north = y - 1 >= 0 ? values[northIdx] : NAN;
    if(current == north && current == west && labels[northIdx] != labels[westIdx])
    {
        int lMin = std::min(labels[northIdx], labels[westIdx]);
        int lMax = std::max(labels[northIdx], labels[westIdx]);

        labels[currentIdx] = lMin;
        mergeLabels(equivalentLabels[lMin], {lMax});
        mergeLabels(equivalentLabels[lMax], {lMin});
    }
    if(current != west && current == north)
    {
        labels[currentIdx] = labels[northIdx];
    }
    if(current != west && current != north)
    {
        labelMax++;
        labels[currentIdx] = labelMax;
        equivalentLabels[labelMax] = {labelMax};
    }
}

std::ostream& operator<<(std::ostream& stream, const std::map<int, std::vector<int>>& equivalentLabels)
{
    for(auto& it : equivalentLabels)
    {
        int key = it.first;
        auto& labels = it.second;
        stream << key << " -> {";
        for(int i = 0; i < labels.size(); i++)
        {
            stream << labels[i];
            if(i < labels.size() - 1)
                stream << ", ";
        }
        stream << "}" << std::endl;
    }
    return stream;
}

int getMinEquivalent(int label, std::map<int, std::vector<int>>& equivalentLabels)
{
    auto eqLabel = equivalentLabels[label].front();
    if(eqLabel == label)
        return label;

    return getMinEquivalent(eqLabel, equivalentLabels);
};

void CCL::apply(const cv::Mat& input, cv::Mat& output)
{
    int width = input.cols;
    int height = input.rows;
    const float* const values = (float*)input.data;

    if(output.empty() || true)
    {
        output = std::move(cv::Mat(height, width, CV_32SC1, cv::Scalar(0))); // filled with background
    }
    int* const labels = (int*)output.data;

    std::map<int, std::vector<int>> equivalentLabels;
    int labelMax = -1;
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            labelPixel(x, y, width, height, values, labels, labelMax, equivalentLabels);
        }
    }

    for(int i = 0; i < width * height; i++)
    {
        labels[i] = getMinEquivalent(labels[i], equivalentLabels);
    }
}
