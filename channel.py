from flask import Flask, request, jsonify
import json
import requests
import time
import random
import re
from better_profanity import profanity
from textblob import TextBlob
from flask_cors import CORS

# Class-based application configuration
class ConfigClass(object):
    """ Flask application config """
    SECRET_KEY = 'This is an INSECURE secret!! DO NOT use this in production!!'

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(__name__ + '.ConfigClass')  # configuration
app.app_context().push()  # create an app context before initializing db

# Configuration
HUB_URL = 'http://127.0.0.1:5555'
#HUB_URL = "http://vm146.rz.uni-osnabrueck.de/hub"
HUB_AUTHKEY = 'Crr-K24d-2N'
CHANNEL_AUTHKEY = '0987654321'
CHANNEL_NAME = "AI Web Discussion Channel"
CHANNEL_ENDPOINT = 'http://127.0.0.1:5000'
#CHANNEL_ENDPOINT = "http://vm146.rz.uni-osnabrueck.de/u065/channel.wsgi"
CHANNEL_FILE = 'messages.json'
CHANNEL_TYPE_OF_SERVICE = 'aiweb24:chat'
MAX_MESSAGES = 50  # Limit to 50 messages
MESSAGE_EXPIRY_SECONDS = 86400000  # 1 day in seconds

# Load better-profanity filter
profanity.load_censor_words()

@app.cli.command('register')
def register_command():
    global CHANNEL_AUTHKEY, CHANNEL_NAME, CHANNEL_ENDPOINT

    # send a POST request to server /channels
    response = requests.post(HUB_URL + '/channels', headers={'Authorization': 'authkey ' + HUB_AUTHKEY},
                             data=json.dumps({
                                "name": CHANNEL_NAME,
                                "endpoint": CHANNEL_ENDPOINT,
                                "authkey": CHANNEL_AUTHKEY,
                                "type_of_service": CHANNEL_TYPE_OF_SERVICE,
                             }))
    if response.status_code == 200:
        print("Success creating channel: "+str(response.status_code))
        print(response.text)
        return
    
    if response.status_code != 200:
        print("Error creating channel: "+str(response.status_code))
        print(response.text)
        return


ALLOWED_TOPICS = [
    # Core AI Concepts
    "artificial intelligence", "machine learning", "deep learning", "neural networks", 
    "computer vision", "natural language processing", "AI ethics", "reinforcement learning", 
    "robotics", "supervised learning", "unsupervised learning", "semi-supervised learning",
    "self-supervised learning", "transfer learning", "zero-shot learning", "one-shot learning",
    "few-shot learning", "multi-modal AI", "explainable AI", "symbolic AI",

    # Large Language Models (LLMs) and NLP
    "chatbots", "LLM", "transformer models", "ChatGPT", "GPT", "GPT-3", "GPT-4", "GPT-4 Turbo", "BERT", "RoBERTa",
    "XLNet", "T5", "BLOOM", "OPT", "Mistral", "LLaMA", "Claude", "Gemini", "PaLM", "Dolly", 
    "open-source LLMs", "text generation", "text summarization", "question answering",
    "named entity recognition", "sentiment analysis", "text classification", "word embeddings",
    "attention mechanism", "tokenization", "vector embeddings", "retrieval-augmented generation (RAG)", 
    "prompt engineering", "fine-tuning", "context length", "in-context learning",

    # AI Methodologies & Training Techniques
    "stochastic gradient descent", "backpropagation", "gradient boosting", "decision trees",
    "support vector machines", "k-means clustering", "random forests", "bayesian networks",
    "genetic algorithms", "autoencoders", "generative adversarial networks (GANs)", 
    "variational autoencoders (VAEs)", "convolutional neural networks (CNNs)", "recurrent neural networks (RNNs)",
    "long short-term memory (LSTM)", "gated recurrent units (GRUs)", "transformers", "attention is all you need",
    "self-attention", "contrastive learning", "meta-learning", "federated learning", "swarm intelligence",

    # AI Applications
    "speech recognition", "speech-to-text", "text-to-speech", "AI music generation", "AI-generated art",
    "autonomous vehicles", "recommendation systems", "AI in healthcare", "AI in finance", "fraud detection",
    "AI in cybersecurity", "AI in gaming", "AI in education", "AI in robotics", "AI-powered assistants",
    "AI in supply chain", "AI in marketing", "AI in customer service", "AI-powered search engines",
    "AI-driven analytics", "AI in drug discovery", "digital twins", "edge AI", "AI-generated video",

    # AI Ethics, Bias & Safety
    "AI bias", "algorithmic fairness", "responsible AI", "explainability", "black box AI", 
    "AI governance", "AI safety", "AI risks", "AI misinformation", "AI for good", 
    "data privacy in AI", "AI hallucinations", "alignment problem", "value alignment",
    "AI regulatory policies", "AI transparency", "model interpretability", "adversarial AI",
    "AI misuse prevention", "truthful AI", "data leakage in AI",

    # AI Tools & Frameworks
    "TensorFlow", "PyTorch", "Keras", "Hugging Face", "scikit-learn", "XGBoost", "LightGBM",
    "ONNX", "JAX", "DeepSpeed", "Ray", "LangChain", "ChromaDB", "OpenAI API", "Anthropic API",
    "Google Vertex AI", "Microsoft Azure AI", "AWS SageMaker", "Colab", "Jupyter Notebooks",

    # AI Trends & Research Topics
    "neuromorphic computing", "AGI", "artificial general intelligence", "AI singularity",
    "AI-powered creativity", "AI-generated code", "AI and quantum computing", "AI in physics",
    "AI in biology", "neural architecture search", "transformer efficiency", "model compression",
    "distillation in deep learning", "quantization in AI", "low-rank adaptation (LoRA)", 
    "RLHF (reinforcement learning from human feedback)", "human-AI collaboration", 
    "multimodal learning", "AI in law", "AI in environmental science", "AI for accessibility"
]


# AI facts for auto-response
AI_FACTS = [
    "Artificial Intelligence can analyze large datasets faster than humans.",
    "The first AI program was written in 1951 by Christopher Strachey.",
    "Machine learning is a subset of AI that allows computers to learn from data.",
    "AI is used in self-driving cars, healthcare, and finance.",
    "Deep learning is a powerful AI technique inspired by the human brain."
]

def check_authorization(request):
    
    print("Expected Authorization:", 'authkey ' + CHANNEL_AUTHKEY)
    print("Received Authorization:", request.headers.get('Authorization'))
    
    if 'Authorization' not in request.headers or request.headers['Authorization'] != 'authkey ' + CHANNEL_AUTHKEY:
        return False
    return True

def is_message_on_topic(content):
    """Check if the message contains AI-related topics or math expressions."""
    content_lower = content.lower()

    # Allow greetings and general phrases
    general_phrases = ["hello", "hi", "hey", "help", "question", "info", "hallo", "hi", "salut", "interesting", "what is that", "agree"]

    # Allow simple mathematical expressions (e.g. "2 + 3", "10 / 2")
    math_pattern = re.compile(r'^\d+\s*[\+\-\*\/]\s*\d+$')

    # Check AI-related topics, greetings, or math operations
    if any(topic in content_lower for topic in ALLOWED_TOPICS) or \
       any(phrase in content_lower for phrase in general_phrases) or \
       math_pattern.match(content_lower):
        return True

    return False


def analyze_sentiment(content):
    """Perform sentiment analysis on the message."""
    analysis = TextBlob(content)
    if analysis.sentiment.polarity > 0.3:
        return "positive"
    elif analysis.sentiment.polarity < -0.3:
        return "negative"
    return "neutral"

@app.route('/health', methods=['GET'])
def health_check():
    if not check_authorization(request):
        return "Invalid authorization", 400
    return jsonify({'name': CHANNEL_NAME}), 200

@app.route('/', methods=['GET'])
def home_page():
    if not check_authorization(request):
        return "Invalid authorization", 400
    return jsonify(read_messages())

@app.route('/', methods=['POST'])
def send_message():
    if not check_authorization(request):
        return jsonify({"error": "Invalid authorization"}), 400

    message = request.json
    
       # If no timestamp is provided, use current time
    if 'timestamp' not in message:
        message['timestamp'] = time.time()
    
    if not message or 'content' not in message or 'sender' not in message:
        return jsonify({"error": "Invalid message format"}), 400

    # Custom profanity filter
    CUSTOM_PROFANITY_LIST = ["spam", "scam", "fake"]
    if any(word in message['content'].lower() for word in CUSTOM_PROFANITY_LIST):
        return jsonify({"error": "Message contains inappropriate content"}), 400

    # Check for profanity using better-profanity library
    if profanity.contains_profanity(message['content']):
        return jsonify({"error": "Message contains inappropriate content"}), 400

    # Check if message is off-topic
    if not is_message_on_topic(message['content']):
        return jsonify({"error": "Message is off-topic. Please discuss AI-related topics."}), 400

    # Analyze sentiment
    sentiment = analyze_sentiment(message['content'])

    # Process extra field and add default values
    extra_data = message.get('extra', {})
    extra_data['sentiment'] = sentiment

    # Add message to list
    messages = read_messages()
    messages.append({
        'content': message['content'],
        'sender': message['sender'],
        'timestamp': time.time(),
        'extra': extra_data,
    })

    # Limit messages based on max count
    messages = filter_old_messages(messages)

    save_messages(messages)
    return jsonify({"message": "Message received", "sentiment": sentiment}), 200


@app.route('/auto_response', methods=['POST'])
def auto_response():
    """ Generate automatic response based on input message. """
    if not check_authorization(request):
        return "Invalid authorization", 400

    incoming_message = request.json.get('content', '').lower()
    response = auto_reply_logic(incoming_message)
    return jsonify({"response": response}), 200


def auto_reply_logic(message):
    if re.search(r'\b(hello|hi|hey)\b', message, re.IGNORECASE):
        return "Hello! Welcome to AI Web Discussion. How can I assist you?"
    
    if "help" in message:
        return "Ask me anything about AI, or type 'fact' for an AI-related fact!"
    
    if "fact" in message:
        return random.choice(AI_FACTS)
    
    # Handling AI-related topics
    if "deep learning" in message:
        return "Deep learning is a powerful AI technique inspired by the human brain."

    if "machine learning" in message:
        return "Machine learning allows computers to learn patterns from data and make predictions."

    if "robotics" in message:
        return "Robotics combines engineering and AI to create machines that can perform human-like tasks."

    if "natural language processing" in message or "nlp" in message:
        return "NLP is a field of AI that helps computers understand and interpret human language."

    if "computer vision" in message:
        return "Computer vision is an AI field that enables computers to interpret and analyze visual data from the world."

    if "ai ethics" in message:
        return "AI ethics focuses on ensuring AI systems are used responsibly, fairly, and without bias."

    return "Thank you for your message. Stay tuned for AI discussions or type 'help' for options."

@app.route('/', methods=['PATCH'])
def update_message():
    if not check_authorization(request):
        return jsonify({"error": "Invalid authorization"}), 400

    patch_data = request.json
    if not patch_data or 'timestamp' not in patch_data:
        return jsonify({"error": "No timestamp provided"}), 400

    messages = read_messages()
    message_found = False

    for message in messages:
        if message['timestamp'] == patch_data['timestamp']:
            # Update the message data; for example, updating the 'extra' field
            if 'extra' in patch_data:
                message['extra'] = patch_data['extra']
            message_found = True
            break

    if not message_found:
        return jsonify({"error": "Message not found"}), 404

    save_messages(messages)
    return jsonify({"message": "Message updated"}), 200



def read_messages():
    #try:
    with open(CHANNEL_FILE, 'r') as f:
        messages = json.load(f)
    #except (FileNotFoundError, json.decoder.JSONDecodeError):
        #messages = []
    return messages

def save_messages(messages):
    with open(CHANNEL_FILE, 'w') as f:
        json.dump(messages, f)

def filter_old_messages(messages):
    """Remove messages exceeding time limit or max count"""
    current_time = time.time()
    messages = [msg for msg in messages if current_time - msg['timestamp'] < MESSAGE_EXPIRY_SECONDS]
    return messages[-MAX_MESSAGES:]  # Keep latest messages

# Preload welcome message
if not read_messages():
    save_messages([{
        'content': "Welcome to the AI Web Discussion Channel! Let's talk AI.",
        'sender': "System",
        'timestamp': time.time(),
        'extra': None,
    }])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    #app.run(debug=False)