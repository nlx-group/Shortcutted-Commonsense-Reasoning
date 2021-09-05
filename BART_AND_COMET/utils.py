from transformers import BartForConditionalGeneration


def extract_bart_model_weights(comet_weights, output_path):
    bart = BartForConditionalGeneration.from_pretrained(comet_weights)
    bart.model.save_pretrained(output_path)
