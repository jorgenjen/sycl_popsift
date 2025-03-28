#pragma once

#include <sycl/sycl.hpp>

#include <memory>
#include <mutex>

namespace popsift {

// class QueueManager
// {
//   public:
//     sycl::queue _device_queue;
//
//     // Delete copy/move constructors and assignment
//     // QueueManager(const QueueManager&) = delete;
//     // QueueManager& operator=(const QueueManager&) = delete;
//     // QueueManager(QueueManager&&) = delete;
//     // QueueManager& operator=(QueueManager&&) = delete;
//
//     // Return by reference (singletons shouldn't return pointers)
//     static QueueManager& getInstance()
//     {
//         static std::once_flag init_flag;
//         static std::unique_ptr<QueueManager> instance;
//
//         // Thread-safe one-time initialization
//         std::call_once(init_flag, []() { instance.reset(new QueueManager()); });
//
//         return *instance;
//     }
//
//   private:
//     QueueManager()
//     {
//         try
//         {
//             sycl::device dev = sycl::device{sycl::gpu_selector_v};
//             _device_queue = sycl::queue(sycl::context{dev}, dev);
//         }
//         catch(sycl::exception const& ex)
//         {
//             std::cout << "No GPU found, falling back to CPU... Exception: " << ex.what() << std::endl;
//             sycl::device dev = sycl::device{sycl::cpu_selector_v};
//             _device_queue = sycl::queue(sycl::context{dev}, dev);
//         }
//     }
//
//     ~QueueManager() = default; // Private destructor
// };

//

class QueueManager
{
  public:
    // sycl::queue _device_queue; // give direct access (afraid of copies)
    // Only thread safe instantiation not sure if I
    // need threadsafe usage of queue if so usage must
    // be mutexed aswell but I believe that is not
    // needed for queue but could be wrong

    // std::shared_ptr<sycl::queue> _device_queue; // not sure if shared is needed
    sycl::queue _device_queue; // not sure if shared is needed

    // TODO: Look into refactoring the usage of posix threads and use sycl::queue
    // to launch the functions as kernels that run on host side and uses the
    // device queue to send everything... probs not needed I thnk

  private:
    // static std::unique_ptr<QueueManager> _instance;
    static QueueManager* _instance;

    static std::mutex mtx;

    // Private contructor instance created in getInstance
    QueueManager()
    {
        try
        {
            // If there is no GPU it will throw exception and use CPU in catch
            sycl::device dev = sycl::device{sycl::gpu_selector_v};
            // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
            _device_queue = sycl::queue(sycl::context{dev}, dev);
            // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
        }
        catch(sycl::exception const& ex)
        {
            // Could set a compile flag and check against it and only show expection
            // when the user has not specified that gpu=off or something like that
            // when doing cmake.. ${flags}

            std::cout << "No GPU found falling back to CPU... Exception thrown: " << ex.what() << std::endl;

            sycl::device dev = sycl::device{sycl::cpu_selector_v};
            // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
            _device_queue = sycl::queue(sycl::context{dev}, dev);
            // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
        }
    }

    // ~QueueManager() { fprintf(stderr, "\n\n\tQueueManager destructor have been called\n"); }

  public:
    QueueManager(const QueueManager& obj) = delete; // no copies

    static QueueManager* getInstance()
    {
        if(_instance == nullptr)
        {
            std::lock_guard<std::mutex> lock(mtx);
            if(_instance == nullptr)
            {
                fprintf(stderr, "\n\tCreating the instance\n");
                _instance = new QueueManager();
                // _instance.reset(new QueueManager());
            }
        }
        return _instance;
    }

    // static QueueManager& getInstance()
    // {
    //     static std::once_flag init_flag;
    //     static std::unique_ptr<QueueManager> instance;
    //
    //     // Thread-safe one-time initialization
    //     std::call_once(init_flag, []() { instance.reset(new QueueManager()); });
    //
    //     return *instance;
    // }
};

} // namespace popsift

// Could make this into a singleton
// But as it's internal and it can only be ran once per PopSift class I don't
// see the need Only thing I need is that everything called in PopSift uses the
// same queue and context Between PopSift instances that does not matter hence
// simple class should be fine
// class QueueManager {
//
//   // TODO: Figure out if mutex is needed to only allow one thread to submit
//   at a
//   // time but don't think that's needed I would believe sycl::queue is thread
//   // safe but could not find it in the documentation
// private:
// };

// I take that back use singleton I thnk...
