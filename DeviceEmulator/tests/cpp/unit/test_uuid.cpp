#include <uuid/uuid.h>

#include <iostream>

int main() {
  // Declare a uuid variable
  uuid_t uuid;

  // Generate a UUID
  uuid_generate(uuid);

  // Convert to string
  char uuid_str[37];
  uuid_unparse(uuid, uuid_str);

  // Print the UUID
  std::cout << "Generated UUID: " << uuid_str << std::endl;

  return 0;
}
