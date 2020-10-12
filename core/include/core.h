#pragma once 

#ifndef EXPORT
#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#endif
#endif

#define OFFSET_ROW_MAJOR(x, y, width, ch) ((ch) * ((y) * (width) + x))
#define OFFSET_COL_MAJOR(x, y, height, ch) ((ch) * ((x) * (height) + y))
#define ALIGN_UP(x, size) ( (x+(size-1))&(~(size-1)) )

constexpr float M_PI = 3.14159265358979323846f;

inline int wrapMirror(int i, int size) {
	if (i < 0) return -i;
	else if (i >= size) return size - (i - size + 1);
	else return i;
}


