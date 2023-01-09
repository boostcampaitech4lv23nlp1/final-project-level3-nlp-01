import os
import azure.cognitiveservices.speech as speechsdk
import time

def base():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    subscription = 'subscription'
    region = 'koreacentral'
    
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))   # region : koreacenreal
    # speech_config = speechsdk.SpeechConfig(subscription=subscription, region=region)   # region : koreacenreal
    speech_config.speech_recognition_language="ko-KR"

    # audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    start_time = time.time()
    audio_config = speechsdk.audio.AudioConfig(filename='./M4a_files/BERT.wav')
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized:\n {}".format(speech_recognition_result.text))
        print("time : ", time.time() - start_time)
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

def dev():
    
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))   # region : koreacenreal
    # speech_config = speechsdk.SpeechConfig(subscription=subscription, region=region)   # region : koreacenreal
    speech_config.speech_recognition_language="ko-KR"

    # audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    start_time = time.time()
    audio_config = speechsdk.audio.AudioConfig(filename='./M4a_files/BERT.wav')
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)


    done = False
    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    # speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))                   # 중간 인식 결과가 포함된 이벤트에 대한 신호.
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt.result.text)))           # 최종 인식 결과가 포함된 이벤트에 대한 신호입니다(성공적인 인식 시도를 나타냄).
    # speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))           # 인식 세션(작업)의 시작을 나타내는 이벤트에 대한 신호입니다.
    # speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))            # 인식 세션(작업)의 끝을 나타내는 이벤트에 대한 신호입니다.
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))                            # 취소된 인식 결과가 포함된 이벤트에 대한 신호입니다. 이러한 결과는 직접 취소 요청으로 인해 취소된 인식 시도를 나타냅니다. 또는 전송 또는 프로토콜 오류를 나타냅니다.
    
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)


    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.5)

    print("end time: ", time.time() - start_time)
    speech_recognizer.stop_continuous_recognition()
    # </SpeechContinuousRecognitionWithFile>

    
    '''
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized:\n {}".format(speech_recognition_result.text))
        print("time : ", time.time() - start_time)
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    '''


if __name__ == '__main__':
    dev()