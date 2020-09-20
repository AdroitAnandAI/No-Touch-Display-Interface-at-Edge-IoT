
import os
import cv2

from queue import Queue 
from threading import Thread, Event

from audio import audio_helper
from speech_library.speech_manager import SpeechManager
from speech_library.speech_proxy import SPEECH_CONFIG


MODELS_DIR = os.path.join(os.path.pardir, 'models')


def stopStream(stream_reader):

    if stream_reader:
        stream_reader.stop_stream()
        stream_reader = None

def received_frames(frames, speech, stt):

    speech.push_data(frames, finish_processing=False)
    utt_text, is_stable = speech.get_result()
    rh_result = utt_text.decode("utf-8").strip().lower()

    if (len(rh_result) > 0):
        print('inside thread: ' + rh_result)
        stt.put(rh_result)



def load_device():
    """Reload audio device list"""
    device_list, default_input_index, loopback_index = \
    							audio_helper.get_input_device_list()

    if not device_list:
        print("No audio devices available")

    return device_list

def detectEvent(utterance, controls, control_syn):

	utters = utterance.split(' ')[-3:]
	utters.reverse()
	print(utters)

	for utter in  utters:
		for control in controls:
			synonyms = control_syn.get(control)
			for synonym in synonyms:
				if synonym in utter:
					print('Event Trigger: ' + control)
					return


def main():

    controls = ['left', 'right', 'up', 'down']

    control_syn = {}
    for control in controls:
    	control_syn.setdefault(control, [])

    control_syn['left'].extend(['let', 'left', 'light', 'live', 'laugh'])
    control_syn['right'].extend(['right', 'write', 'great', 'fight', 'might', 'ride'])
    control_syn['up'].extend(['up', 'hop', 'hope', 'out'])
    control_syn['down'].extend(['down', 'doubt', 'though'])


    device_list = load_device()

    stream_reader = audio_helper.StreamReader(
            			device_list[1][0], received_frames)

    if not stream_reader.initialize():
            print("Failed to initialize Stream Reader")
            speech.close()
            speech = None
            return

    speech = SpeechManager()
    print('speech config = ' + str(SPEECH_CONFIG))
    if not speech.initialize(SPEECH_CONFIG,
            infer_device='CPU',
            batch_size=8):
        print("Failed to initialize ASR recognizer")
        speech.close()
        speech = None
        return

    stt = Queue()

    reading_thread = Thread(target=stream_reader.read_stream, \
    					args=(speech, stt), daemon=True)
    reading_thread.start()


    while (True):

    	# if any sound is deciphered from thread then check last 3 words
        if (stt.qsize() > 0):
        	
        	utterance = stt.get()
        	# print("From Parent: " + utterance)
        	
        	detectEvent(utterance, controls, control_syn)

        k = cv2.waitKey(1) & 0xFF

        # press 'q' to exit
        if k == ord('q'):
            break

    stopStream(stream_reader)
    reading_thread.stop()

if __name__== "__main__" :
	main()