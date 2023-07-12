import gradio as gr
import whisper_at as whisper

link = "https://github.com/YuanGongND/whisper-AT"
text = "[Github]"
paper_link = "https://arxiv.org/pdf/2307.03183.pdf"
paper_text = "[Paper]"

model = whisper.load_model("large-v1")
print('model loaded')

def predict(audio_path, time_resolution):
    def round_time_resolution(time_resolution):
        multiple = float(time_resolution) / 0.4
        rounded_multiple = round(multiple)
        rounded_time_resolution = rounded_multiple * 0.4
        return rounded_time_resolution
    audio_tagging_time_resolution = round_time_resolution(time_resolution)
    result = model.transcribe(audio_path, at_time_res=audio_tagging_time_resolution)
    # ASR Results
    print(result["text"])
    # Audio Tagging Results
    audio_tag_result = whisper.parse_at_label(result, language='follow_asr', top_k=5, p_threshold=-1, include_class_list=list(range(527)))
    print(audio_tag_result)

    asr_output = ""
    for segment in result['segments']:
      asr_output = asr_output + str(segment['start']).zfill(1) + 's-' + str(segment['end']).zfill(1) + 's: ' + segment['text'] + '\n'
    at_output = ""
    for segment in audio_tag_result:
        print(segment)
        at_output = at_output + str(segment['time']['start']).zfill(1) + 's-' + str(segment['time']['end']).zfill(1) + 's: ' + ' ,'.join([x[0] for x in segment['audio tags']]) + '\n'
        print(at_output)
    return asr_output, at_output

iface = gr.Interface(fn=predict,
                    inputs=[gr.Audio(type="filepath", source='microphone'), gr.Textbox(value='10', label='Time Resolution in Seconds (Must be must be an integer multiple of 0.4, e.g., 0.4, 2, 10)')],
                    outputs=[gr.Textbox(label="Speech Output"), gr.Textbox(label="Audio Tag Output")],
                    cache_examples=True,
                    title="Quick Demo of Whisper-AT",
                    description="We are glad to introduce Whisper-AT - A new joint audio tagging and speech recognition model. It outputs background sound labels in addition to text." + f"<a href='{paper_link}'>{paper_text}</a> " + f"<a href='{link}'>{text}</a> <br>" +
                    "Whisper-AT is authored by Yuan Gong, Sameer Khurana, Leonid Karlinsky, and James Glass (MIT & MIT-IBM Watson AI Lab).")
iface.launch(debug=True, share=True)