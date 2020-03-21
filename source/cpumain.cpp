#include <stdio.h>
#include "common.h"

int main()
{
	printf("Hello cpu-slam!\n");
	//Common::LibraryTest();
	Common::Renderer renderer;

	renderer.InitWindow();

	return 0;
}
