import pickle

intents = ["hello", "hi", "hey", "how are you", "bye", "goodbye", "thank you", "thanks", "what is this"]
responses = ["Hello there!", "Hi! How can I help you?", "Hey, nice to see you!", "I'm good, thanks!", "Goodbye!", "See you later!", "You're welcome!", "Glad to help!", "This is EchoMind.ai!"]

with open('intents.pkl', 'wb') as f:
    pickle.dump(intents, f)
with open('responses.pkl', 'wb') as f:
    pickle.dump(responses, f)

print("Models generated!")