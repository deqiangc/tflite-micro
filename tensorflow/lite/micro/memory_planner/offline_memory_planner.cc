/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/memory_planner/offline_memory_planner.h"

namespace tflite {

OfflineMemoryPlanner::OfflineMemoryPlanner(
    const unsigned char* offline_memory_plan, int size) {
  offline_memory_plan_entry_buf_ =
      reinterpret_cast<const OfflineMemoryPlanEntry*>(offline_memory_plan);
  buffer_count_ = size / sizeof(OfflineMemoryPlanEntry);
}

OfflineMemoryPlanner::~OfflineMemoryPlanner() {
  // TODO(deqiangc): Auto-generated destructor stub
}

TfLiteStatus OfflineMemoryPlanner::AddBuffer(
    tflite::ErrorReporter* error_reporter, int size, int first_time_used,
    int last_time_used) {
  TF_LITE_REPORT_ERROR(error_reporter, "Unsupported operation");
  return kTfLiteError;
}

// The largest contiguous block of memory that's needed to hold the layout.
size_t OfflineMemoryPlanner::GetMaximumMemorySize() { return 0; }
// How many buffers have been added to the planner.
int OfflineMemoryPlanner::GetBufferCount() { return buffer_count_; }

TfLiteStatus OfflineMemoryPlanner::GetOffsetForBuffer(
    ErrorReporter* error_reporter, int buffer_index, int* offset) {
  if (buffer_index >= buffer_count_) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "buffer index %d is outside range 0 to %d",
                         buffer_index, buffer_count_);
    return kTfLiteError;
  }
  *offset = offline_memory_plan_entry_buf_[buffer_index].offset;
  return kTfLiteOk;
}

}  // namespace tflite
