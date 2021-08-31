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

#ifndef TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_OFFLINE_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_OFFLINE_MEMORY_PLANNER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_planner/memory_planner.h"

namespace tflite {

class OfflineMemoryPlanner : public MemoryPlanner {
 public:
  OfflineMemoryPlanner(const unsigned char* offline_memory_plan, int size);
  ~OfflineMemoryPlanner() override;

  TfLiteStatus GetOffsetForBuffer(ErrorReporter* error_reporter,
                                  int buffer_index, int* offset) override;
  TfLiteStatus AddBuffer(tflite::ErrorReporter* error_reporter, int size,
                         int first_time_used, int last_time_used) override;

  // The largest contiguous block of memory that's needed to hold the layout.
  size_t GetMaximumMemorySize() override;
  // How many buffers have been added to the planner.
  int GetBufferCount() override;

  // OfflineMemoryPlanner is neither copyable nor movable.
  // OfflineMemoryPlanner(const OfflineMemoryPlanner&) = delete;
  // OfflineMemoryPlanner& operator=(const OfflineMemoryPlanner&) = delete;

 private:
  const OfflineMemoryPlanEntry* offline_memory_plan_entry_buf_;
  int buffer_count_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_OFFLINE_MEMORY_PLANNER_H_
