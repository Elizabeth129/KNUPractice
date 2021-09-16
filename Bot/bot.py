import os
import pickle
import random
import time
import json

import boto3
import numpy as np
import requests
import soundfile as sf
import stanza
import telebot
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pydub import AudioSegment


class SensationAnalysisBot(telebot.TeleBot):
    BUCKET_NAME = "giraffeh-voice-to-text"

    def __init__(self, token, model_name, tokenizer_name, **kwargs):
        super(SensationAnalysisBot, self).__init__(token, kwargs)
        self.model: Model = load_model(f"models/{model_name}")
        with open(f"models/{tokenizer_name}", "rb") as handle:
            self.tokenizer: Tokenizer = pickle.load(handle)

        try:        
            self.translator_client = boto3.client(service_name="translate", region_name="eu-west-1", use_ssl=True)
            self.s3_client = boto3.client(service_name="s3", region_name="eu-west-1", use_ssl=True)
            self.transcribe_client = boto3.client(service_name="transcribe", region_name="eu-west-1", use_ssl=True)
        except:
            pass
        self.sentence_analyzer = stanza.Pipeline('en')
        with open("questions_conf.json", "r") as f:
            self.questions_conf = json.load(f)

        self.chat_evaluations = dict()
        self.chat_used_words = dict()

    def translate_message(self, message_text):
        try:
            return self.translator_client.translate_text(
                Text=message_text,
                SourceLanguageCode='auto',
                TargetLanguageCode='en'
            )
        except:
            return message_text

    def speech_to_text(self, voice_message):
        try:
            file_info = self.get_file(voice_message.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            ogg_file_name = f'{voice_message.file_id}.ogg'
            with open(ogg_file_name, 'wb') as new_file:
                new_file.write(downloaded_file)
            # data, sample_rate = sf.read(ogg_file_name)
            wav_file_name = f'{voice_message.file_id}.wav'
            AudioSegment.from_ogg(ogg_file_name).export(wav_file_name, format='wav')
            # sf.write(wav_file_name, data, sample_rate)
            os.remove(ogg_file_name)
            self.s3_client.upload_file(wav_file_name, SensationAnalysisBot.BUCKET_NAME, wav_file_name)
            os.remove(wav_file_name)
            file_uri = f"https://s3.eu-west-1.amazonaws.com/{SensationAnalysisBot.BUCKET_NAME}/{wav_file_name}"
            job_name = voice_message.file_id
            self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': file_uri},
                # MediaFileFormat='wav',
                IdentifyLanguage=True
            )
            time.sleep(1)
            job = None
            while True:
                job = self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                if job['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                print("Transcription not ready yet...")
                time.sleep(5)
            if job['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
                return None
            try:
                result_json = requests.get(job['TranscriptionJob']['Transcript']['TranscriptFileUri']).json()
                return result_json['results']['transcripts'][0]['transcript']
            except:
                return None
        except:
            return None
        
    def predict_emotion(self, text):
        max_len = self.model.layers[0].input_shape[1]
        model_input = pad_sequences(self.tokenizer.texts_to_sequences([text]), maxlen=max_len, padding="post")
        model_output = sum(np.array(self.model.predict(model_input)[0]) * np.array([0, 1, 2, 3, 4]))
        return (model_output / 4.0) * 100.0

    def reset_chat(self, chat_id):
        try:
            self.chat_evaluations.pop(chat_id)
            self.chat_used_words.pop(chat_id)
        except KeyError:
            pass

    def evaluate_text(self, chat_id, text):
        evaluated = bot.predict_emotion(text)
        cur_score = self.chat_evaluations.get(chat_id)
        if cur_score is None:
            self.chat_evaluations[chat_id] = evaluated
        else:
            self.chat_evaluations[chat_id] = (evaluated + cur_score) / 2.0

    def evaluate_chat(self, chat_id):
        return self.chat_evaluations.get(chat_id)

    def get_next_question(self, chat_id, text):
        doc = self.sentence_analyzer(text)
        candidates = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                word = token.words[0]
                if word.lemma.lower() == "i" or word.lemma.lower() == "you":
                    continue
                if word.lemma.lower() in self.chat_used_words.get(chat_id, dict()):
                    continue
                if word.deprel == "root" and word.upos in ["VERB", "NOUN"]:
                    candidates.append({"word": word, "ner": token.ner})
                if word.deprel == "nmod":
                    candidates.append({"word": word, "ner": token.ner})
                if word.deprel == "obj":
                    candidates.append({"word": word, "ner": token.ner})
                if word.deprel == "nsubj" and "Person=1" not in word.feats:
                    candidates.append({"word": word, "ner": token.ner})
                    
        if len(candidates) == 0:
            possibilities = self.questions_conf["returnings"]
            return random.choice(possibilities)
        candidate_d = random.choice(candidates)
        candidate = candidate_d["word"]
        if chat_id not in self.chat_used_words:
            self.chat_used_words[chat_id] = [candidate.lemma.lower()]
        else:
            self.chat_used_words[chat_id].append(candidate.lemma.lower())
        
        if candidate.upos == "VERB":
            possibilities = list(map(lambda x: x.format(candidate.lemma), self.questions_conf["verb"]))
        elif "PERSON" in candidate_d["ner"]:
            possibilities = list(map(lambda x: x.format(candidate.lemma), self.questions_conf["person"]))
        else:
            possibilities = list(map(lambda x: x.format(candidate.lemma), self.questions_conf["other"]))
        possibilities.extend(self.questions_conf["default"])
        return random.choice(possibilities)


bot = SensationAnalysisBot("1999609936:AAHYkGU_-GbH3PQDifN1-FKxjWXrmIa8RfA",
                           "model_1_6B_set_LSTM_1_hidden_layer",
                           "tokenizer_1_6B_set.pickle")

@bot.message_handler(commands=["start"])
def handle_start_command(message):
    bot.send_message(message.chat.id, "Hey, how are you feeling today?")

@bot.message_handler(commands=["reset"])
def handle_reset_command(message):
    bot.reset_chat(message.chat.id)
    bot.send_message(message.chat.id, "I've forget everything about you!")

@bot.message_handler(commands=["evaluate"])
def handle_evaluate_command(message):
    score = bot.evaluate_chat(message.chat.id)
    if score is None:
        bot.send_message(message.chat.id, "Sorry, I don't know anything about you yet")
    else:
        bot.send_message(message.chat.id, f"You are evaluated to be positive for {score:.2f}% in our conversation")

def process_text(chat_id, text):
    text = bot.translate_message(text)['TranslatedText']
    if (text[0] != '/'):
        bot.evaluate_text(chat_id, text)
        bot.send_message(chat_id, bot.get_next_question(chat_id, text))

@bot.message_handler(content_types=["text"])
def handle_sent_text(message):
    process_text(message.chat.id, message.text)

@bot.message_handler(content_types=["voice"])
def handle_sent_voice(message):
    text = bot.speech_to_text(message.chat.id, message.voice)
    if text is not None:
        process_text(message.chat.id, text)
    else:
        bot.send_message(message.chat.id, "Sorry, I couldn't understand you :(")


if __name__ == '__main__':
    bot.polling()
