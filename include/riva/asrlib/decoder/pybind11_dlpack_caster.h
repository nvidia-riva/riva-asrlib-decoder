/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {
template <>
struct type_caster<DLManagedTensor> {
 public:
 protected:
  DLManagedTensor* value;

 public:
  static constexpr auto name = const_name("dl_managed_tensor");

  template <typename T>
  using cast_op_type = DLManagedTensor*&;

  explicit operator DLManagedTensor* &() { return value; }

  bool load(handle src, bool)
  {
    pybind11::capsule capsule;
    if (pybind11::isinstance<pybind11::capsule>(src)) {
      capsule = pybind11::reinterpret_borrow<pybind11::capsule>(src);
    } else if (pybind11::hasattr(src, "__dlpack__")) {
      // note that, if the user tries to pass in an object with
      // a __dlpack__ attribute instead of a capsule, they have
      // no ability to pass the "stream" argument to __dlpack__

      // this can cause a performance reduction, but not a
      // correctness error, since the default null stream will
      // be used for synchronization if no stream is passed

      // https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.__dlpack__.html

      // I think I'm stealing this. The result of CallMethod
      // should already have a reference count of 1
      capsule = pybind11::reinterpret_steal<pybind11::capsule>(
          PyObject_CallMethod(src.ptr(), "__dlpack__", nullptr));
    } else {
      std::cerr << "pybind11_dlpack_caster.h: not a capsule or a __dlpack__ object" << std::endl;
      return false;
    }

    // is this the same as PyCapsule_IsValid?
    if (strcmp(capsule.name(), "dltensor") != 0) {
      return false;
    }
    value = capsule.get_pointer<DLManagedTensor>();
    capsule.set_name("used_dltensor");
    return true;
  }

  static handle cast(DLManagedTensor* src, return_value_policy /* policy */, handle /* parent */)
  {
    if (src) {
    // why call release here?
    // need to get the capsule a name!
    pybind11:
      capsule capsule(src, reinterpret_cast<void (*)(void*)>(src->deleter));
      capsule.set_name("dltensor");
      return capsule.release();

    } else {
      return pybind11::none().inc_ref();
    }
  }
};
}  // namespace detail
}  // namespace pybind11
