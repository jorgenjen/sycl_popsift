#include "sycl_popsift/common/queue_manager.hpp"

namespace popsift {

// Initialize static members

// sycl::queue _device_queue;
QueueManager* QueueManager::_instance = nullptr;
std::mutex QueueManager::mtx;

} // namespace popsift
