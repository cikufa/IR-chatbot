from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

class ChitChatAPI:
    def __init__(self):
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

    def get_response(self, user_input):
        inputs = self.tokenizer([user_input], return_tensors="pt")
        reply_ids = self.model.generate(**inputs)
        response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    chatter = ChitChatAPI()
    print("Hey Welcome to BlenderBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatter.get_response(user_input)
        print(f"Bot: {response}")
