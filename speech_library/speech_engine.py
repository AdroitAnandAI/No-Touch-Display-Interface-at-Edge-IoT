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
"""Speech Library Engine module"""
import logging

from speech_library.speech_library_wrapper import SpeechLibrary, SPEECH_LIBRARY_SUCCESS, \
    SPEECH_LIBRARY_PARAMETER_INFERENCE_BATCH_SIZE, SPEECH_LIBRARY_PARAMETER_INFERENCE_DEVICE, \
    SPEECH_LIBRARY_RESULT_TYPE_FINAL, SPEECH_LIBRARY_RESULT_TYPE_PREVIEW

_logger = logging.getLogger()


class SpeechLibraryEngine:
    """Speech Library Engine"""
    def __init__(self, asr_lib):
        """
        :param asr_lib: Path to Speech Library DLL or SO
        """
        self._asr_lib = asr_lib
        self._speech = None

    def initialize(self, asr_config, batch_size=None, infer_device=None):
        """Initialize Speech Library and set parameters
        :param asr_config: Path to configuration file
        :param batch_size: Initial batch size
        :param infer_device: Initial inference device
        :return: Status code (SPEECH_LIBRARY_SUCCESS or SPEECH_LIBRARY_ERROR_...)
        """
        self._speech = SpeechLibrary(self._asr_lib)
        res = self._speech.SpeechLibraryInitialize(asr_config)
        if res != SPEECH_LIBRARY_SUCCESS:
            _logger.error("Failed to initialize engine (lib: '%s'). Status code: %d",
                          self._asr_lib, res)
            return res
        if batch_size is not None:
            res = self._speech.SpeechLibrarySetParameter(
                SPEECH_LIBRARY_PARAMETER_INFERENCE_BATCH_SIZE,
                batch_size)
            if res != SPEECH_LIBRARY_SUCCESS:
                _logger.error("Failed to set batch size. Status code: %d", res)
                return res
        if infer_device is not None:
            if infer_device == 'GNA':
                infer_device = 'GNA_AUTO'
            res = self._speech.SpeechLibrarySetParameter(SPEECH_LIBRARY_PARAMETER_INFERENCE_DEVICE,
                                                         infer_device)
            if res != SPEECH_LIBRARY_SUCCESS:
                _logger.error("Failed to set inference device. Status code: %d", res)
                return res
        return SPEECH_LIBRARY_SUCCESS

    def push_data(self, data):
        """Push audio data to Speech Library
        :param data: Audio data to push (bytes)
        :return: True if result is stable (final), False otherwise
        """
        if not self._speech:
            _logger.error("Speech Library not initialized")
            return False

        res, info = self._speech.SpeechLibraryPushData(data)
        _logger.debug("is_result_stable: %d, has_speech_started: %d",
                      info.is_result_stable, info.has_speech_started)

        if res < SPEECH_LIBRARY_SUCCESS:
            _logger.error("Failed to push data. Status code: %d", res)
            return False

        _logger.debug("ASR frames processed: %d", len(data))
        return bool(info.is_result_stable)

    def get_result(self, final=False, finish_processing=False):
        """Get result text from Speech Library
        :param final: Get final result if True, preview result otherwise
        :param finish_processing: Process residue data if True
        :return: Result text (bytes)
        """
        if not self._speech:
            _logger.error("Speech Library not initialized")
            return "[ERROR]"
        if finish_processing:
            res, info = self._speech.SpeechLibraryProcessResidueData()
            if res < SPEECH_LIBRARY_SUCCESS:
                _logger.error("Failed to process residue data. Status code: %d", res)
                return "[ERROR {}]".format(res)
            final = final or info.is_result_stable

        if final:
            result_type = SPEECH_LIBRARY_RESULT_TYPE_FINAL
        else:
            result_type = SPEECH_LIBRARY_RESULT_TYPE_PREVIEW

        res, result = self._speech.SpeechLibraryGetResult(result_type)

        if res < SPEECH_LIBRARY_SUCCESS:
            _logger.error("Failed to get result. Status code: %d", res)
            return "[ERROR {}]".format(res)

        return result

    def close(self):
        """Release Speech Library handle"""
        if self._speech:
            self._speech.SpeechLibraryRelease()
            self._speech = None
