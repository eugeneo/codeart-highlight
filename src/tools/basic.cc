#include "absl/log/log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  LOG(INFO) << "Hello, World!";
  return 0;
}