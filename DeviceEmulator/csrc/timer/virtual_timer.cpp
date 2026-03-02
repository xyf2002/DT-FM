#include "virtual_timer.h"

std::unique_ptr<VirtualTimeManager> virtual_time_manager = nullptr;
std::once_flag kInitVirtualTimeManagerFlag;
