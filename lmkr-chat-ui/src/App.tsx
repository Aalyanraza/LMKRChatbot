import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { readStream, type StreamEvent } from './stream';
import lmkrLogo from './assets/lmkr.png';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources: string[];
  validation: {
    status: 'pending' | 'valid' | 'invalid' | 'none';
    reason?: string;
  };
  isStreaming: boolean;
}

export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: 'msg-' + Date.now(),
      role: 'user',
      text: input,
      sources: [],
      validation: { status: 'none' },
      isStreaming: false,
    };

    const aiMsgId = 'msg-' + (Date.now() + 1);
    const aiPlaceholder: Message = {
      id: aiMsgId,
      role: 'assistant',
      text: '',
      sources: [],
      validation: { status: 'pending' },
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, aiPlaceholder]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.text, user_id: 'demo_user' }),
      });

      await readStream(response, (event: StreamEvent) => {
        setMessages((currentMessages) => {
          const newMessages = [...currentMessages];
          const msgIndex = newMessages.findIndex((m) => m.id === aiMsgId);

          if (msgIndex === -1) return currentMessages;

          const msg = { ...newMessages[msgIndex] };

          if (event.type === 'token' && typeof event.content === 'string') {
            msg.text += event.content;
          } else if (event.type === 'sources' && Array.isArray(event.content)) {
            msg.sources = event.content;
          } else if (event.type === 'status') {
            msg.validation = {
              status: event.is_valid ? 'valid' : 'invalid',
              reason: event.reason,
            };
          } else if (event.type === 'done') {
            msg.isStreaming = false;
            setIsLoading(false);
          }

          newMessages[msgIndex] = msg;
          return newMessages;
        });
      });
    } catch (error) {
      console.error('Stream error:', error);
      setIsLoading(false);
      setMessages((prev) => {
        const last = [...prev];
        if (last[last.length - 1]?.role === 'assistant') {
          last[last.length - 1].text += '\n[Connection Error - Please try again]';
          last[last.length - 1].isStreaming = false;
        }
        return last;
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const renderValidationBadge = (validation: Message['validation']) => {
    const { status } = validation;

    if (status === 'pending') {
      return (
        <div className="badge verifying">
          <div className="spinner"></div>
          <span>Verifying...</span>
        </div>
      );
    }
    if (status === 'valid') {
      return <div className="badge valid">✓ Verified Safe</div>;
    }
    if (status === 'invalid') {
      return <div className="badge invalid">⚠ Possible Inaccuracy</div>;
    }
    return null;
  };

  return (
    <div id="root">
      {/* Header */}
      <div className="header">
        <img className="logo-img" src={lmkrLogo} alt="LMKR logo" />
        <div className="header-center-content">
          <div className="header-title">LMKR</div>
          <div className="header-subtitle">AI Assistant</div>
        </div>
      </div>

      {/* Main Container */}
      <div className="main-container">
        {/* Chat Container */}
        <div className="chat-container" id="chatContainer">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">💭</div>
              <div className="empty-state-text">
                <div className="empty-state-title">Welcome to LMKR</div>
                <div className="empty-state-subtitle">
                  Start a conversation to explore AI-powered insights with verified sources and intelligent validation.
                </div>
              </div>
            </div>
          ) : (
            <div>
              {messages.map((msg) => (
                <div key={msg.id} className={`message-group ${msg.role}`}>
                  <div className={`avatar ${msg.role}`}>{msg.role === 'user' ? '👤' : '🤖'}</div>
                  <div className="message-content">
                    <div className={`message-bubble ${msg.role}`}>
                      {msg.text || (msg.isStreaming ? '✨ Thinking...' : '')}
                    </div>
                    {msg.role === 'assistant' && (
                      <>
                        <div className="message-metadata">
                          {renderValidationBadge(msg.validation)}
                          {msg.sources.length > 0 && (
                            <div className="badge sources">📚 {msg.sources.length} Sources Used</div>
                          )}
                        </div>
                        {msg.validation.status === 'invalid' && msg.validation.reason && (
                          <div className="validation-note">⚠️ {msg.validation.reason}</div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="input-area">
          <form className="input-form" onSubmit={handleSubmit}>
            <div className="input-wrapper">
              <textarea
                ref={textareaRef}
                className="input-field"
                placeholder="Ask me anything..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
              />
            </div>
            <button type="submit" className="send-btn" disabled={isLoading || !input.trim()} title="Send message">
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>

      <style>{`
        @keyframes rotateBg {
          0% { background: linear-gradient(0deg, #0d1b2a 0%, #456586ff 100%); }
          1% { background: linear-gradient(3.6deg, #0d1b2a 0%, #456586ff 100%); }
          2% { background: linear-gradient(7.2deg, #0d1b2a 0%, #456586ff 100%); }
          3% { background: linear-gradient(10.8deg, #0d1b2a 0%, #456586ff 100%); }
          4% { background: linear-gradient(14.4deg, #0d1b2a 0%, #456586ff 100%); }
          5% { background: linear-gradient(18deg, #0d1b2a 0%, #456586ff 100%); }
          6% { background: linear-gradient(21.6deg, #0d1b2a 0%, #456586ff 100%); }
          7% { background: linear-gradient(25.2deg, #0d1b2a 0%, #456586ff 100%); }
          8% { background: linear-gradient(28.8deg, #0d1b2a 0%, #456586ff 100%); }
          9% { background: linear-gradient(32.4deg, #0d1b2a 0%, #456586ff 100%); }
          10% { background: linear-gradient(36deg, #0d1b2a 0%, #456586ff 100%); }
          11% { background: linear-gradient(39.6deg, #0d1b2a 0%, #456586ff 100%); }
          12% { background: linear-gradient(43.2deg, #0d1b2a 0%, #456586ff 100%); }
          13% { background: linear-gradient(46.8deg, #0d1b2a 0%, #456586ff 100%); }
          14% { background: linear-gradient(50.4deg, #0d1b2a 0%, #456586ff 100%); }
          15% { background: linear-gradient(54deg, #0d1b2a 0%, #456586ff 100%); }
          16% { background: linear-gradient(57.6deg, #0d1b2a 0%, #456586ff 100%); }
          17% { background: linear-gradient(61.2deg, #0d1b2a 0%, #456586ff 100%); }
          18% { background: linear-gradient(64.8deg, #0d1b2a 0%, #456586ff 100%); }
          19% { background: linear-gradient(68.4deg, #0d1b2a 0%, #456586ff 100%); }
          20% { background: linear-gradient(72deg, #0d1b2a 0%, #456586ff 100%); }
          21% { background: linear-gradient(75.6deg, #0d1b2a 0%, #456586ff 100%); }
          22% { background: linear-gradient(79.2deg, #0d1b2a 0%, #456586ff 100%); }
          23% { background: linear-gradient(82.8deg, #0d1b2a 0%, #456586ff 100%); }
          24% { background: linear-gradient(86.4deg, #0d1b2a 0%, #456586ff 100%); }
          25% { background: linear-gradient(90deg, #0d1b2a 0%, #456586ff 100%); }
          26% { background: linear-gradient(93.6deg, #0d1b2a 0%, #456586ff 100%); }
          27% { background: linear-gradient(97.2deg, #0d1b2a 0%, #456586ff 100%); }
          28% { background: linear-gradient(100.8deg, #0d1b2a 0%, #456586ff 100%); }
          29% { background: linear-gradient(104.4deg, #0d1b2a 0%, #456586ff 100%); }
          30% { background: linear-gradient(108deg, #0d1b2a 0%, #456586ff 100%); }
          31% { background: linear-gradient(111.6deg, #0d1b2a 0%, #456586ff 100%); }
          32% { background: linear-gradient(115.2deg, #0d1b2a 0%, #456586ff 100%); }
          33% { background: linear-gradient(118.8deg, #0d1b2a 0%, #456586ff 100%); }
          34% { background: linear-gradient(122.4deg, #0d1b2a 0%, #456586ff 100%); }
          35% { background: linear-gradient(126deg, #0d1b2a 0%, #456586ff 100%); }
          36% { background: linear-gradient(129.6deg, #0d1b2a 0%, #456586ff 100%); }
          37% { background: linear-gradient(133.2deg, #0d1b2a 0%, #456586ff 100%); }
          38% { background: linear-gradient(136.8deg, #0d1b2a 0%, #456586ff 100%); }
          39% { background: linear-gradient(140.4deg, #0d1b2a 0%, #456586ff 100%); }
          40% { background: linear-gradient(144deg, #0d1b2a 0%, #456586ff 100%); }
          41% { background: linear-gradient(147.6deg, #0d1b2a 0%, #456586ff 100%); }
          42% { background: linear-gradient(151.2deg, #0d1b2a 0%, #456586ff 100%); }
          43% { background: linear-gradient(154.8deg, #0d1b2a 0%, #456586ff 100%); }
          44% { background: linear-gradient(158.4deg, #0d1b2a 0%, #456586ff 100%); }
          45% { background: linear-gradient(162deg, #0d1b2a 0%, #456586ff 100%); }
          46% { background: linear-gradient(165.6deg, #0d1b2a 0%, #456586ff 100%); }
          47% { background: linear-gradient(169.2deg, #0d1b2a 0%, #456586ff 100%); }
          48% { background: linear-gradient(172.8deg, #0d1b2a 0%, #456586ff 100%); }
          49% { background: linear-gradient(176.4deg, #0d1b2a 0%, #456586ff 100%); }
          50% { background: linear-gradient(180deg, #0d1b2a 0%, #456586ff 100%); }
          51% { background: linear-gradient(183.6deg, #0d1b2a 0%, #456586ff 100%); }
          52% { background: linear-gradient(187.2deg, #0d1b2a 0%, #456586ff 100%); }
          53% { background: linear-gradient(190.8deg, #0d1b2a 0%, #456586ff 100%); }
          54% { background: linear-gradient(194.4deg, #0d1b2a 0%, #456586ff 100%); }
          55% { background: linear-gradient(198deg, #0d1b2a 0%, #456586ff 100%); }
          56% { background: linear-gradient(201.6deg, #0d1b2a 0%, #456586ff 100%); }
          57% { background: linear-gradient(205.2deg, #0d1b2a 0%, #456586ff 100%); }
          58% { background: linear-gradient(208.8deg, #0d1b2a 0%, #456586ff 100%); }
          59% { background: linear-gradient(212.4deg, #0d1b2a 0%, #456586ff 100%); }
          60% { background: linear-gradient(216deg, #0d1b2a 0%, #456586ff 100%); }
          61% { background: linear-gradient(219.6deg, #0d1b2a 0%, #456586ff 100%); }
          62% { background: linear-gradient(223.2deg, #0d1b2a 0%, #456586ff 100%); }
          63% { background: linear-gradient(226.8deg, #0d1b2a 0%, #456586ff 100%); }
          64% { background: linear-gradient(230.4deg, #0d1b2a 0%, #456586ff 100%); }
          65% { background: linear-gradient(234deg, #0d1b2a 0%, #456586ff 100%); }
          66% { background: linear-gradient(237.6deg, #0d1b2a 0%, #456586ff 100%); }
          67% { background: linear-gradient(241.2deg, #0d1b2a 0%, #456586ff 100%); }
          68% { background: linear-gradient(244.8deg, #0d1b2a 0%, #456586ff 100%); }
          69% { background: linear-gradient(248.4deg, #0d1b2a 0%, #456586ff 100%); }
          70% { background: linear-gradient(252deg, #0d1b2a 0%, #456586ff 100%); }
          71% { background: linear-gradient(255.6deg, #0d1b2a 0%, #456586ff 100%); }
          72% { background: linear-gradient(259.2deg, #0d1b2a 0%, #456586ff 100%); }
          73% { background: linear-gradient(262.8deg, #0d1b2a 0%, #456586ff 100%); }
          74% { background: linear-gradient(266.4deg, #0d1b2a 0%, #456586ff 100%); }
          75% { background: linear-gradient(270deg, #0d1b2a 0%, #456586ff 100%); }
          76% { background: linear-gradient(273.6deg, #0d1b2a 0%, #456586ff 100%); }
          77% { background: linear-gradient(277.2deg, #0d1b2a 0%, #456586ff 100%); }
          78% { background: linear-gradient(280.8deg, #0d1b2a 0%, #456586ff 100%); }
          79% { background: linear-gradient(284.4deg, #0d1b2a 0%, #456586ff 100%); }
          80% { background: linear-gradient(288deg, #0d1b2a 0%, #456586ff 100%); }
          81% { background: linear-gradient(291.6deg, #0d1b2a 0%, #456586ff 100%); }
          82% { background: linear-gradient(295.2deg, #0d1b2a 0%, #456586ff 100%); }
          83% { background: linear-gradient(298.8deg, #0d1b2a 0%, #456586ff 100%); }
          84% { background: linear-gradient(302.4deg, #0d1b2a 0%, #456586ff 100%); }
          85% { background: linear-gradient(306deg, #0d1b2a 0%, #456586ff 100%); }
          86% { background: linear-gradient(309.6deg, #0d1b2a 0%, #456586ff 100%); }
          87% { background: linear-gradient(313.2deg, #0d1b2a 0%, #456586ff 100%); }
          88% { background: linear-gradient(316.8deg, #0d1b2a 0%, #456586ff 100%); }
          89% { background: linear-gradient(320.4deg, #0d1b2a 0%, #456586ff 100%); }
          90% { background: linear-gradient(324deg, #0d1b2a 0%, #456586ff 100%); }
          91% { background: linear-gradient(327.6deg, #0d1b2a 0%, #456586ff 100%); }
          92% { background: linear-gradient(331.2deg, #0d1b2a 0%, #456586ff 100%); }
          93% { background: linear-gradient(334.8deg, #0d1b2a 0%, #456586ff 100%); }
          94% { background: linear-gradient(338.4deg, #0d1b2a 0%, #456586ff 100%); }
          95% { background: linear-gradient(342deg, #0d1b2a 0%, #456586ff 100%); }
          96% { background: linear-gradient(345.6deg, #0d1b2a 0%, #456586ff 100%); }
          97% { background: linear-gradient(349.2deg, #0d1b2a 0%, #456586ff 100%); }
          98% { background: linear-gradient(352.8deg, #0d1b2a 0%, #456586ff 100%); }
          99% { background: linear-gradient(356.4deg, #0d1b2a 0%, #456586ff 100%); }
          100% { background: linear-gradient(360deg, #0d1b2a 0%, #456586ff 100%); }
        }

        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        html, body, #root {
          height: 100%;
          width: 100%;
          overflow: hidden;
        }

        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          background: linear-gradient(0deg, #0d1b2a 0%, #456586ff 100%);
          animation: rotateBg 15s linear infinite;
          color: #e2e8f0;
        }

        #root {
          display: flex;
          flex-direction: column;
        }

        /* Header Styles */
        .header {
          background: linear-gradient(180deg, rgba(13, 27, 42, 0.95) 0%, rgba(22, 42, 63, 0.85) 100%);
          backdrop-filter: blur(20px);
          border-bottom: 1px solid rgba(148, 163, 184, 0.1);
          padding: 1rem 2rem;
          display: flex;
          align-items: center;
          justify-content: space-between;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
          position: relative;
        }

        .header-center-content {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .logo-img {
          width: 50px;
          height: 50px;
          border-radius: 10px;
          background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 700;
          color: white;
          font-size: 24px;
          box-shadow: 0 8px 16px rgba(243, 247, 247, 0.3);
          transition: all .7s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .logo-img:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 24px rgba(243, 247, 247, 0.4);
        }

        .header-title {
          font-size: 2rem;
          font-weight: 800;
          background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          letter-spacing: -0.5px;
          margin: 0;
        }

        .header-subtitle {
          font-size: 0.8rem;
          color: #94a3b8;
          font-weight: 400;
          letter-spacing: 0.5px;
          margin: 0;
          margin-top: 12px;
        }

        /* Main Container */
        .main-container {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        /* Chat Area */
        .chat-container {
          flex: 1;
          overflow-y: auto;
          padding: 2rem;
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .chat-container::-webkit-scrollbar {
          width: 8px;
        }

        .chat-container::-webkit-scrollbar-track {
          background: transparent;
        }

        .chat-container::-webkit-scrollbar-thumb {
          background: rgba(148, 163, 184, 0.3);
          border-radius: 4px;
          transition: background 0.3s ease;
        }

        .chat-container::-webkit-scrollbar-thumb:hover {
          background: rgba(148, 163, 184, 0.5);
        }

        /* Empty State */
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          gap: 2rem;
        }

        .empty-state-icon {
          width: 80px;
          height: 80px;
          border-radius: 20px;
          background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 40px;
          animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }

        .empty-state-text {
          text-align: center;
          max-width: 400px;
        }

        .empty-state-title {
          font-size: 1.5rem;
          font-weight: 700;
          margin-bottom: 0.5rem;
        }

        .empty-state-subtitle {
          color: #94a3b8;
          font-size: 0.95rem;
          line-height: 1.6;
        }

        /* Message Group */
        .message-group {
          display: flex;
          gap: 1rem;
          animation: slideUp 0.4s ease-out;
        }

        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .message-group.user {
          justify-content: flex-end;
        }

        .message-group.assistant {
          justify-content: flex-start;
        }

        /* Avatar */
        .avatar {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 20px;
          font-weight: 700;
          flex-shrink: 0;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .avatar.user {
          background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
          color: white;
        }

        .avatar.assistant {
          background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
          color: white;
        }

        /* Message Bubble */
        .message-content {
          max-width: 70%;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .message-bubble {
          padding: 0.75rem 1.5rem;
          border-radius: 16px;
          line-height: 1.6;
          font-size: 0.95rem;
          word-wrap: break-word;
          animation: popIn 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        @keyframes popIn {
          0% {
            opacity: 0;
            transform: scale(0.95);
          }
          100% {
            opacity: 1;
            transform: scale(1);
          }
        }

        .message-bubble.user {
          background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
          color: white;
          border-bottom-right-radius: 4px;
          box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
        }

        .message-bubble.assistant {
          background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
          border: 1px solid rgba(148, 163, 184, 0.2);
          color: #e2e8f0;
          border-bottom-left-radius: 4px;
          backdrop-filter: blur(10px);
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }

        /* Metadata */
        .message-metadata {
          display: flex;
          gap: 0.75rem;
          flex-wrap: wrap;
          align-items: center;
          padding: 0 0.5rem;
          font-size: 0.85rem;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 0.875rem;
          border-radius: 20px;
          font-weight: 500;
          font-size: 0.8rem;
          animation: slideIn 0.3s ease-out 0.1s backwards;
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        .badge.verifying {
          background: rgba(59, 130, 246, 0.15);
          color: #60a5fa;
          border: 1px solid rgba(96, 165, 250, 0.3);
        }

        .badge.valid {
          background: rgba(34, 197, 94, 0.15);
          color: #4ade80;
          border: 1px solid rgba(74, 222, 128, 0.3);
        }

        .badge.invalid {
          background: rgba(239, 68, 68, 0.15);
          color: #f87171;
          border: 1px solid rgba(248, 113, 113, 0.3);
        }

        .badge.sources {
          background: rgba(168, 85, 247, 0.15);
          color: #d8b4fe;
          border: 1px solid rgba(216, 180, 254, 0.3);
        }

        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid currentColor;
          border-top-color: transparent;
          border-radius: 50%;
          animation: spin 0.6s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .validation-note {
          color: #f87171;
          font-size: 0.85rem;
          padding: 0.75rem 1rem;
          background: rgba(239, 68, 68, 0.1);
          border-left: 3px solid #f87171;
          border-radius: 6px;
          margin-top: 0.5rem;
        }

        /* Input Area */
        .input-area {
          padding: 1.5rem 2rem;
          background: linear-gradient(180deg, rgba(15, 23, 42, 0.5) 0%, rgba(30, 41, 59, 0.5) 100%);
          backdrop-filter: blur(20px);
          border-top: 1px solid rgba(148, 163, 184, 0.1);
        }

        .input-form {
          display: flex;
          gap: 1rem;
          align-items: flex-end;
        }

        .input-wrapper {
          flex: 1;
          display: flex;
          align-items: flex-end;
          gap: 0.75rem;
          background: rgba(51, 65, 85, 0.5);
          border: 1px solid rgba(148, 163, 184, 0.2);
          border-radius: 12px;
          padding: 0.75rem 1.25rem;
          transition: all 0.3s ease;
          backdrop-filter: blur(10px);
        }

        .input-wrapper:focus-within {
          border-color: rgba(59, 130, 246, 0.5);
          background: rgba(51, 65, 85, 0.7);
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .input-field {
          flex: 1;
          background: transparent;
          border: none;
          color: #e2e8f0;
          font-size: 0.95rem;
          outline: none;
          font-family: inherit;
          max-height: 120px;
          resize: none;
          length : 1.4
        }

        .input-field::placeholder {
          color: #64748b;
        }

        /* Send Button */
        .send-btn {
          padding: 0.85rem;
          background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
          border: none;
          border-radius: 10px;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
          flex-shrink: 0;
          
          aspect-ratio: 1;
        }

        .send-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
        }

        .send-btn:active:not(:disabled) {
          transform: translateY(0);
        }

        .send-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        /* Responsive */
        @media (max-width: 768px) {
          .message-content {
            max-width: 90%;
          }

          .chat-container {
            padding: 1.5rem;
          }

          .input-area {
            padding: 1rem 1.5rem;
          }

          .header {
            padding: 1rem 1.5rem;
          }

          .message-bubble {
            padding: 1rem 1.25rem;
            font-size: 0.9rem;
          }
        }

        /* Utility */
        .flex {
          display: flex;
        }

        .flex-col {
          flex-direction: column;
        }

        .gap-1 {
          gap: 0.25rem;
        }

        .items-start {
          align-items: flex-start;
        }
      `}</style>
    </div>
  );
}