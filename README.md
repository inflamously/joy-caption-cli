## JoyCLI: An toolkit for image auto captioning and sorting

This repositories wraps the joy caption model from [fancyfeast/joy-caption-alpha-two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two) and just extends it with a cli tool for local usage or on a server.
<br>All model credits go to their respective owners aswell as fancyfeast for the finetune.

---

### Usage
To run this application run the following steps:

#### Install dependencies
```
python -m venv venv
pip install -r requirements.txt
```

#### Example commands
```
clip folder caption "G:\whatever\folder\images" --prompt_prefix "test123" --batch_size 64
joycaption folder caption "<folder_path>" beta --caption_type "Describe" --caption_length "short" --custom_prompt "describe the image in a short three sentence prompt" --batch_size 24 --prompt_prefix <custom_prefix>
joycaption folder quality "<folder_path>" beta --batch_size 16
joycaption folder organize "<folder_path>" beta --batch_size 16
joycaption folder caption "<folder_path>" beta --caption_type "Describe" --caption_length "short" --custom_prompt "<whatever prompt>" --batch_size 24 --prompt_prefix <custom_prefix>
```

### Support

* Saleforce BLIP [https://github.com/salesforce/BLIP](BLIP)
* Joy Caption Alpha Two [huggingface.co/.../fancyfeast/joy-caption-alpha-two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two)
* Joy Caption Beta One [huggingface.co/.../fancyfeast/joy-caption-beta-one](https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one)