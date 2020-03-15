#pragma once

namespace Common
{
	template<class T>
	class Point
	{
	public:
		constexpr Point(T x = { 0 }, T y = { 0 }, T z = { 0 }) : x(x), y(y), z(z) {}

		constexpr Point<T>& operator/=(const T& scalar) {	
			x /= scalar; y /= scalar; z /= scalar; 
		}
	
		T x = { 0 };
		T y = { 0 };
		T z = { 0 };
	};
		
	template<class T>
	constexpr Point<T> operator+(const Point<T>& p1, const Point<T>& p2) {
		return { p1.x + p2.x, p1.y + p2.y, p1.z + p2.z };
	}
}
