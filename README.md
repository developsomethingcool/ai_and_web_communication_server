# AI Web Discussion Chat

A distributed chat system focused on AI-related discussions, built with Flask and React. The system includes a central hub server, multiple chat channels, and different client implementations.

## 🌟 Features

- Distributed architecture with a central hub and multiple chat channels
- Real-time AI-focused discussion channels
- Multiple client implementations (Flask Web Client, React Client)
- Message sentiment analysis
- Profanity filtering
- Topic enforcement for AI-related discussions
- Auto-response system for common AI queries
- Authentication system with channel-specific auth keys

## 🏗️ Architecture

The project consists of four main components:

1. **Hub Server (hub.py)**
   - Central server managing channel registration
   - Maintains channel health checks
   - SQLite database for channel management

2. **Channel Server (channel.py)**
   - Individual chat channel implementation
   - Message filtering and validation
   - Sentiment analysis
   - AI topic enforcement
   - Auto-response system

3. **Web Client (client.py)**
   - Flask-based web interface
   - Channel listing and message display
   - Message posting capabilities

4. **React Client (react_client.py)**
   - Modern React-based UI
   - Real-time message updates
   - Channel search functionality
   - Responsive design with Tailwind CSS

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/developsomethingcool/ai_and_web_communication_server.git
cd ai_and_web_communication_server
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
```

3. Install required packages:
```bash
pip install flask flask-sqlalchemy flask-cors requests textblob better-profanity
```

4. Initialize the database:
```bash
python hub.py
```

## 🎮 Usage

1. Start the hub server:
```bash
python hub.py
```

2. Start a channel server:
```bash
python channel.py
```

3. Start the web client:
```bash
python client.py
```

4. For the React client, open react_client.py in a web browser.

Default ports:
- Hub Server: 5555
- Channel Server: 5000
- Web Client: 5005

## 🔑 Configuration

Key configuration variables are defined in each component:

### Hub Server
```python
SERVER_AUTHKEY = 'Crr-K24d-2N'
SQLALCHEMY_DATABASE_URI = 'sqlite:///chat_server.sqlite'
```

### Channel Server
```python
HUB_URL = 'http://127.0.0.1:5555'
HUB_AUTHKEY = 'Crr-K24d-2N'
CHANNEL_AUTHKEY = '0987654321'
MAX_MESSAGES = 50
MESSAGE_EXPIRY_SECONDS = 86400000
```

### Web Client
```python
HUB_AUTHKEY = 'Crr-K24d-2N'
HUB_URL = 'http://localhost:5555'
```

## 🛡️ Security Features

- Authentication using auth keys
- Message content validation
- Profanity filtering
- Channel health checks
- Input sanitization
- Cross-Origin Resource Sharing (CORS) configuration

## 🤖 AI Features

- Topic enforcement for AI-related discussions
- Sentiment analysis on messages
- Auto-response system for AI queries
- Comprehensive list of allowed AI topics
- AI fact generation

## 🔧 Customization

The system can be customized by modifying:
- `ALLOWED_TOPICS` in channel.py for topic control
- `AI_FACTS` in channel.py for auto-responses
- Sentiment analysis thresholds
- Message expiry and limits
- Authentication keys
- UI components in the React client

## 📝 API Endpoints

### Hub Server
- `GET /channels` - List all channels
- `POST /channels` - Register new channel
- `GET /` - Home page

### Channel Server
- `GET /health` - Channel health check
- `GET /` - Get all messages
- `POST /` - Send message
- `PATCH /` - Update message
- `POST /auto_response` - Get AI-related auto-response

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



