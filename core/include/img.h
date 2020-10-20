#pragma once 

#include <vector>
#include <string>
#include <ostream>
#include <iomanip>

#include "core.h"

enum class ImgType {
    Undefined,
    UInt8,
    Int32,
    Float32
};

enum class ImgStoringMethod {
    RowMajor,
    ColMajor
};

class Img {
public:
    EXPORT Img();
    EXPORT Img(int width, int height, ImgType type, int channels = 1);

    EXPORT Img(const Img& other) = default;
    Img& operator=(const Img& other) = default;

    EXPORT Img(Img&& other) noexcept;
    Img& operator=(Img&& other) noexcept;

    EXPORT Img(uint8_t* data, int width, int height, ImgType type, int channels = 1) noexcept;

    Img EXPORT copyAsBlank(const Img& other, ImgType newType = ImgType::Undefined);

    int channels;
    int width;
    int height;
    ImgType type;

    template<typename T = uint8_t*>
    const T* data() const { return (T*)_data; }
    template<typename T = uint8_t*>
    T* data() { return (T*)_data; }

    bool EXPORT isEmpty() const;
    bool EXPORT isWrapped()const;

    size_t EXPORT elemSize() const;

    size_t EXPORT offset(int x, int y, int channel = 0, ImgStoringMethod storing = ImgStoringMethod::RowMajor) const;

private:
    uint8_t* _data;
    std::vector<uint8_t> _buffer;
};

std::ostream& operator<<(std::ostream& stream, const Img& img) {
    stream << "Img { ";
    stream << &img << ", ";
    stream << img.width << "x" << img.height << "x" << img.channels << ", ";

    switch (img.type)
    {
    case ImgType::UInt8:
        stream << "UInt8";
        break;
    case ImgType::Int32:
        stream << "Int32";
        break;
    case ImgType::Float32:
        stream << "Float32";
        break;
    default:
        stream << "Undefined";
    }
    stream << ", ";

    stream << (size_t)img.width * img.height * img.channels * img.elemSize() << " bytes, ";
    stream << img.data<void*>() << ", ";
    img.isWrapped() ? stream << "wrapped" : stream << "non-wrapped";
    stream << "}";
    return stream;
}