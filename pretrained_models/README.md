# Pretrained Weights

The Whisper-AT script downloads the original OpenAI Whisper model and our AT model automatically. So you do not really need to download it manually. 
But in case your device does not have Internet access, here is the link. Download and place the model in the same directory as the original OpenAI Whisper model. 

These links support `wget`.

```python
dropbox_path = [
"tiny.en": "https://www.dropbox.com/s/atq9so6w0qug5ai/tiny.en_ori.pth?dl=1",
"tiny": "https://www.dropbox.com/s/06f2h29aki39q9r/tiny_ori.pth?dl=1",
"base.en": "https://www.dropbox.com/s/qtzgsbuquoz0afn/base.en_ori.pth?dl=1",
"base": "https://www.dropbox.com/s/4vn2oatda321y7h/base_ori.pth?dl=1",
"small.en": "https://www.dropbox.com/s/cyx50ycl1ul7lji/small.en_ori.pth?dl=1",
"small.en_low": "https://www.dropbox.com/s/507o66zgl8v6ddd/small.en_low.pth?dl=1",
"small": "https://www.dropbox.com/s/5zqzs3e47zwhjd3/small_ori.pth?dl=1",
"small_low": "https://www.dropbox.com/s/3lxlmh437tneifl/small_low.pth?dl=1",
"medium.en": "https://www.dropbox.com/s/bbvylvmgns8ja4p/medium.en_ori.pth?dl=1",
"medium.en_low": "https://www.dropbox.com/s/2q5wprr8f9gti5t/medium.en_low.pth?dl=1",
"medium": "https://www.dropbox.com/s/93zfj4afmv0qfyl/medium_ori.pth?dl=1",
"medium_low": "https://www.dropbox.com/s/g66h1vtn1u426dj/medium_low.pth?dl=1",
"large-v1": "https://www.dropbox.com/s/b8x2en1fdzc8nhk/large-v1_ori.pth?dl=1",
"large-v1_low": "https://www.dropbox.com/s/5o79h70wyla8jlk/large-v1_low.pth?dl=1",
"large-v2": "https://www.dropbox.com/s/94x7wqw4hscpls0/large-v2_ori.pth?dl=1",
"large-v2_low": "https://www.dropbox.com/s/wk5dyxustpji06c/large-v2_low.pth?dl=1",
"large": "https://www.dropbox.com/s/94x7wqw4hscpls0/large-v2_ori.pth?dl=1",
"large_low": "https://www.dropbox.com/s/wk5dyxustpji06c/large-v2_low.pth?dl=1"]
```

## China Mirror Links

The models are hosted on Dropbox. If dropbox is not accessible, use a VPN or the mirror link, you would have to donwload it manually and place it in the same directory (by default `~/.cache/whisper`) as the original OpenAI Whisper model.
[[镜像链接(腾讯微云)]](https://share.weiyun.com/bVxQWxTe)