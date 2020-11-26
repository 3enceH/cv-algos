#include <ostream>
#include <cassert>

#include "core.h"
#include "cudaenv.h"

namespace custd {
    template<typename Type, size_t Capacity>
    class vector {
    public:
        typedef Type* iterator;
        typedef const Type* const_iterator;

        __host__ vector() : _size(0) {
            void* devPtr;
            CHECK_CUDA(cudaMalloc(&devPtr, sizeof(Type) * Capacity));
            _elems = (Type*)devPtr;
            CHECK_CUDA(cudaMemset((void*)_elems, 0, sizeof(Type) * Capacity));
            printf("vector allocate %p capacity %llu elemsize %llu bytes %llu\n", _elems, Capacity, sizeof(Type), sizeof(Type) * Capacity);
        }

        __host__ ~vector() {
            if (_elems != nullptr) {
                printf("vector free %p capacity %llu elemsize %llu bytes %llu\n", _elems, Capacity, sizeof(Type), sizeof(Type) * Capacity);
                CHECK_CUDA(cudaFree(_elems));
                _elems = nullptr;
            }
        }

        __host__ vector(const vector& other) = default;
        __host__ vector& operator=(const vector& other) = default;

        __host__ vector(vector&& other) noexcept {
            _size = other._size;
            _elems = other._elems;

            other._size = 0;
            other._elems = nullptr;
        }

        __host__ vector& operator=(vector&& other) noexcept
        {
            _size = other._size;
            _elems = other._elems;

            other._size = 0;
            other._elems = nullptr;

            return *this;
        }

        __device__ void push_back(const Type& value) {
            assert(_size <= Capacity);
            _elems[_size] = value;
            _size++;
        }

        __device__ void pop_back() {
            _size--;
        }

        __device__ iterator erase(const_iterator position) {
            assert(position >= begin() && position <= end());
            if (position == end()) return end();
            return erase(position, position + 1);
        }

#define NON_CONST(baseptr, cptr)  ((baseptr) + ((cptr) - (baseptr)))

        __device__ iterator erase(const_iterator first, const_iterator last) {
            assert(first >= begin() && last <= end());
            if (first > last || last - first == 0) {
                return end();
            }
            if (last == end()) {
                _size = first - begin();
                return end();
            }
            iterator from = NON_CONST(begin(), last);
            iterator to = NON_CONST(begin(), first);
            while (from != end()) {
                // swap, must save pointer to allocated mem
                std::swap(*to, *from);
                ++from;
                ++to;
                assert(from <= begin() + Capacity);
                assert(to <= begin() + Capacity);
            }
            _size = _size - (last - first);
            return NON_CONST(begin(), first); // const_it to it
        }

        __device__ Type& operator[](size_t idx) {
            assert(idx < _size);
            return _elems[idx];
        }
        __device__ const Type& operator[](size_t idx) const { return this->operator[](idx); }

        __device__ iterator begin() { return &_elems[0]; }
        __device__ const_iterator begin() const { return &_elems[0]; }
        __device__ iterator end() { return &_elems[_size]; }
        __device__ const_iterator end() const { return &_elems[_size]; }

        __device__ bool empty() const { return _size == 0; }
        __device__ size_t size() const { return _size; }
        __device__ size_t capacity() const { return Capacity; }

        __device__ Type* data() { return _elems; }
        __device__ const Type* data() const { return _elems; }

        __device__ void clear() {
            _size = 0;
        }

    private:
        Type* _elems;
        size_t _size;
    };

    template<typename Type, size_t Capacity>
    std::ostream& operator<<(std::ostream& stream, const vector<Type, Capacity>& v) {
        stream << "vector[";
        for (auto it = v.begin(); it != v.end(); ++it) {
            stream << *it;
            if (it != v.end() - 1) stream << ", ";
        }
        stream << "]";
        return stream;
    }

    template<typename It>
    std::ostream& print(std::ostream& stream, It first, It last) {
        stream << "[";
        for (auto it = first; it != last; ++it) {
            stream << *it;
            if (it != last - 1) stream << ", ";
        }
        stream << "]";
        return stream;
    }

    template<typename Key, typename Value>
    struct pair {
        Key first;
        Value second;

        __host__ pair(const pair<Key, Value>& other) = default;
        __host__ pair<Key, Value>& operator=(const pair<Key, Value>& other) = default;

        bool operator<(const pair<Key, Value>& other) { return this->first < other.first; }
        bool operator==(const pair<Key, Value>& other) { return this->first == other.first; }
        bool operator<=(const pair<Key, Value>& other) { return *this < other || *this == other; }
        bool operator>(const pair<Key, Value>& other) { return !(*this <= other); }
    };

    template<typename Key, typename Value>
    std::ostream& operator<<(std::ostream& stream, const pair<Key, Value>& p) {
        stream << "pair{\"" << p.first<<"\": " << p.second << "}";
        return stream;
    }

//#define kernel_throw_error asm("trap;");

    template<typename Key, typename Value, size_t Capacity>
    class map {
    public:
        typedef pair<Key, Value>* iterator;
        typedef const pair<Key, Value>* const_iterator;

        __host__ map() : _size(0) {
            void* devPtr;
            CHECK_CUDA(cudaMalloc(&devPtr, sizeof(pair<Key, Value>) * Capacity));
            _elems = (pair<Key, Value>*)devPtr;

            uint8_t buffer[sizeof(pair<Key, Value>) * Capacity];
            pair<Key, Value>* hostRepr = (pair<Key, Value>*)buffer;
            for (int i = 0; i < Capacity; i++) {
                hostRepr[i].second = Value();
            }
            CHECK_CUDA(cudaMemcpy((void*)_elems, (void*)buffer, sizeof(pair<Key, Value>) * Capacity, cudaMemcpyKind::cudaMemcpyHostToDevice));
            printf("map allocate %p capacity %llu elemsize %llu bytes %llu\n", _elems, Capacity, sizeof(pair<Key, Value>), sizeof(pair<Key, Value>) * Capacity);
        }
        __host__ ~map() {
            uint8_t buffer[sizeof(pair<Key, Value>) * Capacity];
            CHECK_CUDA(cudaMemcpy((void*)buffer, (void*)_elems, sizeof(pair<Key, Value>) * Capacity, cudaMemcpyKind::cudaMemcpyDeviceToHost));
            pair<Key, Value>* hostRepr = (pair<Key, Value>*)buffer;
            for (int i = 0; i < Capacity; i++) {
                hostRepr[i].second.~Value();
            }
            if (_elems != nullptr) {
                printf("map free %p capacity %llu elemsize %llu bytes %llu\n", _elems, Capacity, sizeof(pair<Key, Value>), sizeof(pair<Key, Value>) * Capacity);
                CHECK_CUDA(cudaFree(_elems));
                _elems = nullptr;
            }
        }

        __device__ Value& operator[](Key key) {
            auto it = find(key);
            if (it == end()) {
                auto it = _insert(key);
                return it->second;
            }
            return it->second;
        }
        __device__ const Value& operator[](Key key) const {
            auto it = find(key);
            assert(it != end());
            return it->second;
        }

        __device__ iterator erase(const_iterator position) {
            assert(position >= begin() && position <= end());
            if (position == end()) return end();
            return erase(position, position + 1);
        }

        __device__ iterator erase(const_iterator first, const_iterator last) {
            assert(first >= begin() && last <= end());
            if (first > last || last - first == 0) {
                return end();
            }
            if (last == end()) {
                _size = first - begin();
                return end();
            }
            iterator from = NON_CONST(begin(), last);
            iterator to = NON_CONST(begin(), first);
            while (from != end()) {
                // swap, must save pointer to allocated mem
                std::swap(*to, *from);
                ++from;
                ++to;
                assert(from <= begin() + Capacity);
                assert(to <= begin() + Capacity);
            }
            _size = _size - (last - first);
            return NON_CONST(begin(), first); // const_it to it
        }

        __device__ size_t erase(Key key) {
            size_t s = size();
            auto it = find(key);
            if (it == end()) return 0;
            erase(it);
            return s - size();
        }

        __device__ iterator find(Key key) {
            for (auto it = begin(); it != end(); ++it) {
                if (key == it->first) return it;
            }
            return end();
        }

        __device__ iterator begin() { return &_elems[0]; }
        __device__ const_iterator begin() const { return &_elems[0]; }
        __device__ iterator end() { return &_elems[_size]; }
        __device__ const_iterator end() const{ return &_elems[_size]; }

        __device__ bool empty() const { return _size == 0; }
        __device__ size_t size() const { return _size; }
        __device__ size_t capacity() const { return Capacity; }

        __device__ void clear() {
            _size = 0;
        }

    private:
        __device__ iterator _insert(Key key) {
            iterator it = nullptr;
            if (_size < Capacity) {
                _elems[_size].first = key;
                _size++;

                //TODO search and insert instead of sort
                sort(begin(), end());
                return find(key);
            }
            return it;
        }

        pair<Key, Value>* _elems;
        size_t _size;
    };

    template<typename Key, typename Value, size_t Capacity>
    std::ostream& operator<<(std::ostream& stream, const map<Key, Value, Capacity>& v) {
        stream << "map{";
        for (auto it = v.begin(); it != v.end(); ++it) {
            stream << "\"" << it->first << "\": ";
            stream << it->second;
            if (it != v.end() - 1) stream << ", ";
        }
        stream << "}";
        return stream;
    }

    template<typename Type>
    inline void swap(Type& a, Type& b) { 

        uint8_t buffer[sizeof(Type)];
        memcpy(buffer, &a, sizeof(Type));
        memcpy(&b, &a, sizeof(Type));
        memcpy(&b, &buffer, sizeof(Type));
    }

    template<typename It>
    void sort(It first, It last) {
        int counter = 0;
        assert(last >= first);
        // yepp, da bubble
        bool swapped = true;
        for (int i = 0; i < last - first && swapped; i++) {
            swapped = false;
            for (auto it = first + 1; it != last - i; ++it) {
                if (*(it - 1) > *it)
                {
                    std::swap(*(it - 1), *it);
                    swapped = true;
                }
            }
        }
    }
}
