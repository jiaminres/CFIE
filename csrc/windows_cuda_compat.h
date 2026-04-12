#pragma once

#ifdef _WIN32
#include <c10/util/win32-headers.h>

#ifdef small
#undef small
#endif

// Linux toolchains often provide this alias implicitly; MSVC does not.
using uint = unsigned int;
#endif
