#pragma once 

#ifndef EXPORT
#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#endif
#endif

#define OFFSET_ROW_MAJOR(x, y, colSize, channels) ((channels) * ((y) * (colSize) + (x)))
#define OFFSET_COL_MAJOR(x, y, rowSize, channels) ((channels) * ((x) * (rowSize) + (y)))

#define BORDER_REFLECT(idx, size) (((idx) < 0) ? (-(idx) - 1) : ((idx) >= size) ? (size) - ((idx) - (size) + 1) : (idx))
#define BORDER_REFLECT_101(idx, size) (((idx) < 0) ? (-(idx)) : ((idx) >= size) ? (size) - ((idx) - (size) + 2) : (idx))
#define ALIGN_UP(x, size) ( (x+(size-1))&(~(size-1)) )
#define CLAMP(val, min, max) ((val) < (min) ? (min) : (val) > (max) ? (max) : (val))

#define XSTR(a) #a
#define STR(a) XSTR(a)

constexpr float M_PI = 3.14159265358979323846f;

