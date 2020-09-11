# *****************************************************************************
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
# *****************************************************************************
"""Speech Library Wrapper module"""
import ctypes
import functools

# SpeechLibraryParameter enum values
SPEECH_LIBRARY_PARAMETER_INFERENCE_BATCH_SIZE = 0
SPEECH_LIBRARY_PARAMETER_INFERENCE_DEVICE = 1

# SpeechLibraryResultType enum values
SPEECH_LIBRARY_RESULT_TYPE_PARTIAL = 0
SPEECH_LIBRARY_RESULT_TYPE_PREVIEW = 1
SPEECH_LIBRARY_RESULT_TYPE_FINAL = 2

# SpeechLibraryStatus enum values
SPEECH_LIBRARY_SUCCESS = 0
SPEECH_LIBRARY_ERROR_GENERIC = -1
SPEECH_LIBRARY_ERROR_OUT_OF_MEMORY = -2
SPEECH_LIBRARY_ERROR_INVALID_RESOURCE = -4
SPEECH_LIBRARY_ERROR_INVALID_PARAM = -5
SPEECH_LIBRARY_ERROR_INVALID_HANDLE_VALUE = -6
SPEECH_LIBRARY_ERROR_INVALID_STATE = -9

SpeechLibraryHandle = ctypes.c_void_p


class SpeechLibraryProcessingInfo(ctypes.Structure):
    """Processing Info structure"""

    _fields_ = [
        ('is_result_stable', ctypes.c_int32),
        ('has_speech_started', ctypes.c_int32),
    ]

    _pack_ = 4


def validate_handle(function):
    """Decorator for convenient validation of Speech Library handle"""
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.is_speech_handle_valid():
            return function(self, *args, **kwargs)

        raise ValueError("Speech Library is not initialized")

    return wrapper


class SpeechLibrary:
    """Speech Library"""

    def __init__(self, speech_lib):
        """
        :param speech_lib: Path to Speech Library DLL or SO
        """
        self._dll = ctypes.CDLL(speech_lib)
        self._speech_handle = SpeechLibraryHandle()
        self._SpeechLibraryCreate()
        self._info = SpeechLibraryProcessingInfo()
        self._result_buffer = ctypes.create_string_buffer(1024 * 1024)

    def _SpeechLibraryCreate(self):
        """Create Speech Library instance
        :return: Status code (SPEECH_LIBRARY_SUCCESS or SPEECH_LIBRARY_ERROR_...)
        """
        res = self._dll.SpeechLibraryCreate(
            ctypes.byref(self._speech_handle)
        )
        return res

    @validate_handle
    def SpeechLibraryInitialize(self, configuration_filename):
        """Initialize Speech Library
        :param configuration_filename: Path to configuration file
        :return: Status code (SPEECH_LIBRARY_SUCCESS or SPEECH_LIBRARY_ERROR_...)
        """
        res = self._dll.SpeechLibraryInitialize(
            self._speech_handle,
            ctypes.c_char_p(configuration_filename.encode('utf-8'))
        )
        return res

    @validate_handle
    def SpeechLibrarySetParameter(self, parameter, value):
        """Set parameter of Speech Library
        :param parameter: Parameter ID (SPEECH_LIBRARY_PARAMETER_...)
        :param value: Value of the parameter (device name as string or batch size as int)
        :return: Status code (SPEECH_LIBRARY_SUCCESS or SPEECH_LIBRARY_ERROR_...)
        """
        if isinstance(value, str):
            c_value = ctypes.create_string_buffer(value.encode('utf-8'))
        else:
            c_value = ctypes.c_uint32(value)

        res = self._dll.SpeechLibrarySetParameter(
            self._speech_handle,
            parameter,
            ctypes.byref(c_value),
            ctypes.sizeof(c_value)
        )
        return res

    @validate_handle
    def SpeechLibraryPushData(self, data):
        """Push audio data to Speech Library
        :param data: Audio data to push (bytes)
        :return: Tuple of status code and Processing Info structure
        """
        short_data = ctypes.cast(data, ctypes.POINTER(ctypes.c_short))
        res = self._dll.SpeechLibraryPushData(
            self._speech_handle,
            short_data,
            len(data) // 2,
            ctypes.byref(self._info)
        )
        return res, self._info

    @validate_handle
    def SpeechLibraryProcessResidueData(self):
        """Process residue data
        :return: Tuple of status code and Processing Info structure
        """
        res = self._dll.SpeechLibraryProcessResidueData(
            self._speech_handle,
            ctypes.byref(self._info)
        )
        return res, self._info

    @validate_handle
    def SpeechLibraryGetResult(self, result_type):
        """Get result text from Speech Library
        :param result_type: Result type
        :return: Tuple of status code and result text (bytes)
        """
        res = self._dll.SpeechLibraryGetResult(
            self._speech_handle,
            result_type,
            self._result_buffer,
            len(self._result_buffer)
        )
        result = ctypes.string_at(self._result_buffer)
        return res, result

    @validate_handle
    def SpeechLibraryReset(self):
        """Reset Speech Library
        :return: Status code (SPEECH_LIBRARY_SUCCESS or SPEECH_LIBRARY_ERROR_...)
        """
        res = self._dll.SpeechLibraryReset(
            self._speech_handle
        )
        return res

    @validate_handle
    def SpeechLibraryRelease(self):
        """Release Speech Library handle
        :return: Status code (SPEECH_LIBRARY_SUCCESS or SPEECH_LIBRARY_ERROR_...)
        """
        res = self._dll.SpeechLibraryRelease(
            ctypes.byref(self._speech_handle)
        )
        return res

    def is_speech_handle_valid(self):
        """Return True if Speech Library handle is valid, False otherwise"""
        return bool(self._speech_handle.value)
