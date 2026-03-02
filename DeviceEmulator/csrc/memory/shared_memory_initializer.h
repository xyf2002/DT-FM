#ifndef SHARED_MEMORY_INITIALIZER_H
#define SHARED_MEMORY_INITIALIZER_H

#include <stddef.h>

#include "shared_memory_manager.h"

/**
 * @brief Initializes the shared memory with the given size.
 *
 * @param shm_size The size of the shared memory to be initialized (in bytes).
 * @return shared_memory_t* A pointer to the initialized shared memory
 * structure, or NULL if initialization fails.
 */
shared_memory_t* initialize_shared_memory(size_t shm_size);

#endif  // SHARED_MEMORY_INITIALIZER_H
