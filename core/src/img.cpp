#include "img.h"

Img::Img() : width(0), height(0), type(ImgType::Undefined), channels(0), _data(nullptr) {}

Img::Img(int width, int height, ImgType type, int channels /* = 1*/)
    : width(width), height(height), type(type), channels(channels)
{
    _buffer.resize(width * height * channels * elemSize());
    _data = _buffer.data();
}

Img::Img(Img&& other) noexcept
{
    width = other.width;
    height = other.height;
    channels = other.channels;
    type = other.type;

    _buffer = std::move(other._buffer);
    _data = _buffer.data();
}

Img& Img::operator=(Img&& other) noexcept
{
    width = other.width;
    height = other.height;
    channels = other.channels;
    type = other.type;

    _buffer = std::move(other._buffer);
    _data = _buffer.data();
    return *this;
}

Img::Img(uint8_t* data, int width, int height, ImgType type, int channels /* = 1*/) noexcept
{
    this->width = width;
    this->height = height;
    this->type = type;
    this->channels = channels;
    this->_data = data;
}

Img Img::copyAsBlank(const Img& other, ImgType newType /* = ImgType::Undefined*/)
{
    if(newType == ImgType::Undefined)
        return std::move(Img(width, height, type, channels));
    else
        return std::move(Img(width, height, newType, channels));
}

int channels;
int width;
int height;
ImgType type;

bool Img::isEmpty() const
{
    return type == ImgType::Undefined;
}
bool Img::isWrapped() const
{
    return type != ImgType::Undefined && _buffer.empty();
}

size_t Img::elemSize() const
{
    switch(type)
    {
    case ImgType::UInt8:
        return sizeof(uint8_t);
    case ImgType::Int32:
        return sizeof(int32_t);
    case ImgType::Float32:
        return sizeof(float);
    default:
        return -1;
    }
}

size_t Img::offset(int x, int y, int channel /* = 0*/, ImgStoringMethod storing /* = ImgStoringMethod::RowMajor*/) const
{
    if(storing == ImgStoringMethod::RowMajor)
        return (channel + 1) * (y * width + x);
    else if(storing == ImgStoringMethod::ColMajor)
        return (channel + 1) * (x * height + y);
    else
        return -1;
}
