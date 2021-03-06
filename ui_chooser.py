import enquiries

def choose(options, label_formatter = lambda x: x, prompt = "Choose one of these options: "):
    if len(options) == 0:
        return None
        
    option_labels = [f"[{index + 1}] {label_formatter(obj)}"[:100] for (index, obj) in enumerate(options)]
    chosen_label = enquiries.choose(prompt, option_labels)
    return options[option_labels.index(chosen_label)]