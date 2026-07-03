import { useState, useRef, useEffect } from 'react'
import './App.css'
 
// Replace with your actual API URL from `terraform output api_url`
const API_URL = 'https://9kpo3z1a87.execute-api.us-east-1.amazonaws.com/chat'
 
function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm a UMD chatbot. Ask me about courses, professors, gen-ed requirements, admissions, dining — anything UMD!",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)
 
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])
 
  const sendMessage = async () => {
    if (!input.trim() || loading) return
   
    const userMessage = input.trim()
    const newMessages = [...messages, { role: 'user', content: userMessage }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)
   
    try {
      // Build the history Lambda expects — just role + content, no extra metadata
      const history = newMessages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content }))
   
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          history: history.slice(0, -1)  // everything BEFORE the new message
        }),
      })
   
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`)
      }
   
      const data = await response.json()
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.reply,
          intent: data.intent,
          source: data.source,
        },
      ])
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Sorry, something went wrong: ${error.message}`,
          isError: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }
 
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }
 
  const suggestedPrompts = [
    'What humanities classes are 3 credits?',
    'Is Dave Mount a good professor?',
    'How do I apply to UMD?',
    'Tell me about CMSC131',
  ]
 
  return (
<div className="app">
<header className="header">
<h1>UMD AI Assistant</h1>
<p className="subtitle">Ask me anything about the University of Maryland</p>
</header>
 
      <main className="chat-container">
<div className="messages">
          {messages.map((msg, i) => (
<div key={i} className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}>
<div className="bubble">
                {msg.content}
                {msg.source && (
<div className="meta">
                    intent: {msg.intent} · source: {msg.source}
</div>
                )}
</div>
</div>
          ))}
          {loading && (
<div className="message assistant">
<div className="bubble loading">Thinking...</div>
</div>
          )}
<div ref={messagesEndRef} />
</div>
 
        {messages.length <= 1 && (
<div className="suggestions">
<p>Try asking:</p>
<div className="suggestion-buttons">
              {suggestedPrompts.map((p, i) => (
<button key={i} onClick={() => setInput(p)} className="suggestion">
                  {p}
</button>
              ))}
</div>
</div>
        )}
 
        <div className="input-area">
<input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question..."
            disabled={loading}
            maxLength={200}
          />
<button onClick={sendMessage} disabled={loading || !input.trim()}>
            Send
</button>
</div>
</main>
 
      <footer className="footer">
        Built with AWS Bedrock, OpenSearch, DynamoDB, and Lambda
</footer>
</div>
  )
}
 
export default App