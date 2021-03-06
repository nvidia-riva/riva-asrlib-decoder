"""
Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

# Each platform creates a constraint @<platform>//:platform_constraint that
# is listed in its constraint_values; rule that want to select a specific
# platform to run on can put @<platform>//:platform_constraing into their
# exec_compatible_with attribute.
# Toolchains can similarly be marked with target_compatible_with or
# exec_compatible_with to bind them to this platform.
constraint_setting(
    name = "platform_setting"
)

constraint_value(
    name = "platform_constraint",
    constraint_setting = ":platform_setting",
    visibility = ["//visibility:public"],
)

platform(
    name = "platform",
    visibility = ["//visibility:public"],
    constraint_values = [
        "@bazel_tools//platforms:%{cpu}",
        "@bazel_tools//platforms:%{platform}",
        ":platform_constraint",
    ],
    exec_properties = %{exec_properties},
)
