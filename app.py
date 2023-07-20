import gradio as gr
import whisper_at as whisper

link = "https://github.com/YuanGongND/whisper-AT"
text = "[Github]"
paper_link = "https://arxiv.org/pdf/2307.03183.pdf"
paper_text = "[Paper]"

model_large = whisper.load_model("large-v1")
model_tiny = whisper.load_model("tiny")
model_tiny_en = whisper.load_model("tiny.en")
model_small = whisper.load_model("small")

mdl_dict = {"tiny": model_tiny, "tiny.en": model_tiny_en, "small": model_small, "large": model_large}
lan_dict = {"English": 'en', "Chinese": 'zh'}

def round_time_resolution(time_resolution):
    multiple = float(time_resolution) / 0.4
    rounded_multiple = round(multiple)
    rounded_time_resolution = rounded_multiple * 0.4
    return rounded_time_resolution

def predict(audio_path_m, audio_path_t, model_size, language, time_resolution):
    # print(audio_path_m, audio_path_t)
    # print(type(audio_path_m), type(audio_path_t))
    #return audio_path_m, audio_path_t
    if ((audio_path_m is None) != (audio_path_t is None)) == False:
        return "Please upload and only upload one recording, either upload the audio file or record using microphone.", "Please upload and only upload one recording, either upload the audio file or record using microphone."
    else:
        audio_path = audio_path_m or audio_path_t
        audio_tagging_time_resolution = round_time_resolution(time_resolution)
        model = mdl_dict[model_size]
        if language == 'Auto Detection':
            result = model.transcribe(audio_path, at_time_res=audio_tagging_time_resolution)
        else:
            result = model.transcribe(audio_path, at_time_res=audio_tagging_time_resolution, language=lan_dict[language])
        audio_tag_result = whisper.parse_at_label(result, language='follow_asr', top_k=5, p_threshold=-1, include_class_list=list(range(527)))
        asr_output = ""
        for segment in result['segments']:
          asr_output = asr_output + format(segment['start'], ".1f") + 's-' + format(segment['end'], ".1f") + 's: ' + segment['text'] + '\n'
        at_output = ""
        for segment in audio_tag_result:
            print(segment)
            at_output = at_output + format(segment['time']['start'], ".1f") + 's-' + format(segment['time']['end'], ".1f") + 's: ' + ', '.join([x[0] for x in segment['audio tags']]) + '\n'
            print(at_output)
        return asr_output, at_output

iface = gr.Interface(fn=predict,
                    inputs=[gr.Audio(type="filepath", source='microphone', label='Please either upload an audio file or record using the microphone.', show_label=True), gr.Audio(type="filepath"),
                            gr.Radio(["tiny", "tiny.en", "small", "large"], value='large', label="Model size", info="The larger the model, the better the performance and the slower the speed."),
                            gr.Radio(["Auto Detection", "English", "Chinese"], value='Auto Detection', label="Language", info="Please specify the language, or let the model detect it automatically"),
                            gr.Textbox(value='10', label='Time Resolution in Seconds (Must be must be an integer multiple of 0.4, e.g., 0.4, 2, 10)')],
                    outputs=[gr.Textbox(label="Speech Output"), gr.Textbox(label="Audio Tag Output")],
                    cache_examples=True,
                    title="Quick Demo of Whisper-AT",
                    description="We are glad to introduce Whisper-AT - A new joint audio tagging and speech recognition model. It outputs background sound labels in addition to text." + f"<a href='{paper_link}'>{paper_text}</a> " + f"<a href='{link}'>{text}</a> <br>" +
                    "Whisper-AT is authored by Yuan Gong, Sameer Khurana, Leonid Karlinsky, and James Glass (MIT & MIT-IBM Watson AI Lab). It is an Interspeech 2023 paper.")
iface.launch(debug=True, share=True)