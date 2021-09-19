#ifndef ASSERT
#define ASSERT

#define assert(x) {\
	if (!(x)) {\
		int q = 0 / 0;\
	}\
}

#endif