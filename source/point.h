#pragma once

#include <glm/glm.hpp>

namespace Common
{
	template<class T>
	class Point
	{
	public:
		constexpr Point(T x = { 0 }, T y = { 0 }, T z = { 0 }) : x(x), y(y), z(z) {}

		constexpr Point<T>(const glm::vec3& vector) : x(vector.x), y(vector.y), z(vector.z) {}

		constexpr Point<T>& operator/=(const T& scalar) {	
			x /= scalar; y /= scalar; z /= scalar; 
		}

		constexpr operator glm::vec3() const {
			return glm::vec3(x, y, z);
		}

		constexpr T Length() const {
			return std::sqrt(x * x + y * y + z * z);
		}
		
		constexpr static Point<T> Zero() {
			return Point<T>(); 
		}

		constexpr static Point<T> One() {
			return Point<T>({ 1 }, { 1 }, { 1 });
		}

	
		T x = { 0 };
		T y = { 0 };
		T z = { 0 };

	};
		
	template<class T>
	constexpr Point<T> operator+(const Point<T>& p1, const Point<T>& p2) {
		return { p1.x + p2.x, p1.y + p2.y, p1.z + p2.z };
	}

	template<class T>
	constexpr Point<T> operator-(const Point<T>& p1, const Point<T>& p2) {
		return { p1.x - p2.x, p1.y - p2.y, p1.z - p2.z };
	}
}
