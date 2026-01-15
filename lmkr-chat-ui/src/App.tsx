import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, X, PhoneOff } from 'lucide-react';
import { readStream, type StreamEvent } from './stream';
import lmkrLogo from './assets/lmkr.png';
import ReactMarkdown from 'react-markdown';
import { 
  LiveKitRoom, 
  RoomAudioRenderer, 
  useLocalParticipant,
  useConnectionState,
  useSpeakingParticipants
} from '@livekit/components-react';
import { ConnectionState } from 'livekit-client';
import '@livekit/components-styles';

// --- Interfaces ---
interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources: string[];
  isStreaming: boolean;
}

// --- Custom Components ---

// 1. The Visualizer Orb (Revolving Circle)
type AgentState = 'listening' | 'speaking' | 'thinking' | 'disconnected';

// Updated Visualizer Orb accepting state
const VoiceOrb = ({ state }: { state: AgentState }) => {
  return (
    <div className={`orb-container ${state}`}>
      <div className="orb-ring-outer"></div>
      <div className="orb-ring-inner"></div>
      <div className="orb-core"></div>
    </div>
  );
};

const VoiceSession = ({ onDisconnect }: { onDisconnect: () => void }) => {
  const connectionState = useConnectionState();
  
  // Hook that returns an array of everyone currently speaking
  const activeSpeakers = useSpeakingParticipants();

  // Check if any remote person (the AI Agent) is speaking
  const isAgentSpeaking = activeSpeakers.some(p => !p.isLocal);
  
  // Check if you (the local user) are speaking
  const isUserSpeaking = activeSpeakers.some(p => p.isLocal);
  
  const [agentState, setAgentState] = useState<AgentState>('disconnected');

  useEffect(() => {
    if (connectionState !== ConnectionState.Connected) {
      setAgentState('disconnected');
      return;
    }

    if (isAgentSpeaking) {
      setAgentState('speaking');
    } else if (isUserSpeaking) {
      setAgentState('listening');
    } else {
      setAgentState('thinking');
    }
  }, [connectionState, isAgentSpeaking, isUserSpeaking]);

  const getStatusText = () => {
    switch (agentState) {
      case 'speaking': return 'Agent Speaking';
      case 'listening': return 'Listening...';
      case 'thinking': return 'Thinking...';
      default: return 'Connecting...';
    }
  };

  return (
    <div className="voice-agent-interface">
      <div className="voice-header-status">
        <span className={`live-indicator ${connectionState === ConnectionState.Connected ? 'active' : ''}`}></span> 
        Live Session
      </div>

      <div className="visualizer-area">
        <VoiceOrb state={agentState} />
        <div className={`agent-status-text ${agentState}`}>
          {getStatusText()}
        </div>
      </div>

      <RoomAudioRenderer />
      
      <CustomVoiceControls onDisconnect={onDisconnect} />
    </div>
  );
};

// 2. Custom Minimal Controls (Mic & Hangup only)
const CustomVoiceControls = ({ onDisconnect }: { onDisconnect: () => void }) => {
  const { localParticipant } = useLocalParticipant();
  const [isMuted, setIsMuted] = useState(false);

  const toggleMute = () => {
    if (localParticipant) {
      const newMutedState = !isMuted;
      localParticipant.setMicrophoneEnabled(!newMutedState);
      setIsMuted(newMutedState);
    }
  };

  return (
    <div className="custom-voice-controls">
      <button 
        className={`control-btn ${isMuted ? 'muted' : ''}`} 
        onClick={toggleMute}
        title={isMuted ? "Unmute" : "Mute"}
      >
        {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
      </button>
      
      <button 
        className="control-btn hangup" 
        onClick={onDisconnect}
        title="End Call"
      >
        <PhoneOff size={24} />
      </button>
    </div>
  );
};

// --- Main App Component ---

export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [voiceChatActive, setVoiceChatActive] = useState(false);
  const [voiceToken, setVoiceToken] = useState<string>('');
  const [voiceUrl, setVoiceUrl] = useState<string>('');
  const [isLoadingVoice, setIsLoadingVoice] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bufferRef = useRef<string>('');
  const displayedLengthRef = useRef<number>(0);
  const intervalRef = useRef<number | null>(null);
  const currentMsgIdRef = useRef<string>('');

  const handleStartVoiceChat = async () => {
    try {
      setIsLoadingVoice(true);
      const randomId = 'user_' + Math.floor(Math.random() * 10000);
      const response = await fetch(`http://localhost:8000/get_token?user_id=${randomId}`);
      const data = await response.json();
      
      if (data.token && data.url) {
        setVoiceToken(data.token);
        setVoiceUrl(data.url);
        setVoiceChatActive(true);
      }
    } catch (error) {
      console.error('Failed to start voice chat:', error);
      alert('Failed to connect to voice chat. Please try again.');
    } finally {
      setIsLoadingVoice(false);
    }
  };

  const handleEndVoiceChat = () => {
    setVoiceChatActive(false);
    setVoiceToken('');
    setVoiceUrl('');
  };

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

  // Smooth streaming logic (kept as is)
  const startSmoothDisplay = (msgId: string) => {
    if (intervalRef.current) return;
    
    intervalRef.current = window.setInterval(() => {
      const buffer = bufferRef.current;
      const displayedLength = displayedLengthRef.current;
      
      if (displayedLength < buffer.length) {
        const chunkSize = Math.min(2, buffer.length - displayedLength);
        const newDisplayedLength = displayedLength + chunkSize;
        const textToShow = buffer.substring(0, newDisplayedLength);
        
        setMessages((currentMessages) => {
          const newMessages = [...currentMessages];
          const msgIndex = newMessages.findIndex((m) => m.id === msgId);
          if (msgIndex !== -1) {
            newMessages[msgIndex] = { ...newMessages[msgIndex], text: textToShow };
          }
          return newMessages;
        });
        
        displayedLengthRef.current = newDisplayedLength;
      }
    }, 20);
  };

  const stopSmoothDisplay = (msgId: string) => {
    const checkComplete = () => {
      if (displayedLengthRef.current >= bufferRef.current.length) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        setMessages((currentMessages) => {
          const newMessages = [...currentMessages];
          const msgIndex = newMessages.findIndex((m) => m.id === msgId);
          if (msgIndex !== -1) {
            newMessages[msgIndex] = { 
              ...newMessages[msgIndex], 
              isStreaming: false 
            };
          }
          return newMessages;
        });
      } else {
        setTimeout(checkComplete, 50);
      }
    };
    checkComplete();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: 'msg-' + Date.now(),
      role: 'user',
      text: input,
      sources: [],
      isStreaming: false,
    };

    const aiMsgId = 'msg-' + (Date.now() + 1);
    const aiPlaceholder: Message = {
      id: aiMsgId,
      role: 'assistant',
      text: '',
      sources: [],
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, aiPlaceholder]);
    setInput('');
    setIsLoading(true);
    
    bufferRef.current = '';
    displayedLengthRef.current = 0;
    currentMsgIdRef.current = aiMsgId;

    try {
      const response = await fetch('http://localhost:8000/chat_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.text, user_id: 'demo_user' }),
      });

      await readStream(response, (event: StreamEvent) => {
        if (event.type === 'token' && typeof event.content === 'string') {
          bufferRef.current += event.content;
          if (!intervalRef.current) {
            startSmoothDisplay(aiMsgId);
          }
        } else if (event.type === 'sources' && Array.isArray(event.content)) {
          setMessages((currentMessages) => {
            const newMessages = [...currentMessages];
            const msgIndex = newMessages.findIndex((m) => m.id === aiMsgId);
            if (msgIndex !== -1) {
              newMessages[msgIndex] = { ...newMessages[msgIndex], sources: event.content as string[] };
            }
            return newMessages;
          });
        } else if (event.type === 'done') {
          stopSmoothDisplay(aiMsgId);
          setIsLoading(false);
        }
      });
    } catch (error) {
      console.error('Stream error:', error);
      stopSmoothDisplay(aiMsgId);
      setIsLoading(false);
      setMessages((prev) => {
        const last = [...prev];
        if (last[last.length - 1]?.role === 'assistant') {
          last[last.length - 1].text = bufferRef.current + '\n[Connection Error - Please try again]';
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

  return (
    <div id="root">
      {/* Header */}
      <div className="header">
        <img className="logo-img" src={lmkrLogo} alt="LMKR logo" />
        <div className="header-center-content">
          <div className="header-title">LMKR</div>
          <div className="header-subtitle">AI Assistant</div>
        </div>
        <button 
          onClick={handleStartVoiceChat}
          disabled={isLoadingVoice}
          className="voice-btn"
          title="Start voice chat"
        >
          <Mic size={20} />
        </button>
      </div>

      {/* Main Container */}
      <div className="main-container">
        {/* Chat Container */}
        <div className="chat-container" id="chatContainer">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">💬</div>
              <div className="empty-state-text">
                <div className="empty-state-title">Welcome to LMKR</div>
                <div className="empty-state-subtitle">
                  Start a conversation to explore AI-powered insights with verified sources.
                </div>
              </div>
            </div>
          ) : (
            <div>
              {messages.map((msg) => (
                <div key={msg.id} className={`message-group ${msg.role}`}>
                  <div className={`avatar ${msg.role}`}>
                    {msg.role === 'user' ? '👤' : <img src={lmkrLogo} alt="LMKR" className="avatar-logo" />}
                  </div>
                  <div className="message-content">
                    <div className={`message-bubble ${msg.role}`}>
                      {msg.text ? <ReactMarkdown>{msg.text}</ReactMarkdown> : (msg.isStreaming ? <div className="typing-indicator"><span></span><span></span><span></span></div> : '')}
                    </div>
                    {msg.role === 'assistant' && (
                      <>
                        <div className="message-metadata">
                          {msg.sources.length > 0 && (
                            <div className="badge sources">📚 {msg.sources.length} Sources Used</div>
                          )}
                        </div>
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

      {/* Modern Voice Chat Modal */}
      {/* Modern Voice Chat Modal */}
      {voiceChatActive && voiceToken && voiceUrl && (
        <div className="voice-chat-overlay">
          <div className="voice-chat-modal">
            <LiveKitRoom
              serverUrl={voiceUrl}
              token={voiceToken}
              connect={true}
              audio={true}
              video={false}
              data-lk-theme="default"
              style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            >
              {/* REPLACED THE OLD MANUAL CONTENT WITH THE SMART COMPONENT */}
              <VoiceSession onDisconnect={handleEndVoiceChat} />
            </LiveKitRoom>
          </div>
        </div>
      )}

      <style>{`
        /* --- ORB ANIMATIONS --- */
        .voice-agent-interface {
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: space-between;
          padding: 2rem;
          color: white;
          background: radial-gradient(circle at center, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0) 70%);
        }

        .voice-header-status {
          font-size: 0.9rem;
          color: #94a3b8;
          display: flex;
          align-items: center;
          gap: 8px;
          text-transform: uppercase;
          letter-spacing: 1px;
          font-weight: 600;
        }

        .live-indicator {
          width: 8px;
          height: 8px;
          background-color: #ef4444;
          border-radius: 50%;
          box-shadow: 0 0 10px #ef4444;
          animation: pulse-red 2s infinite;
        }

        .visualizer-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 2rem;
        }

        .orb-container {
          position: relative;
          width: 200px;
          height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        /* The Core */
        .orb-core {
          position: absolute;
          width: 100px;
          height: 100px;
          border-radius: 50%;
          background: radial-gradient(circle at 30% 30%, rgba(96, 165, 250, 0.2), rgba(37, 99, 235, 0.6));
          box-shadow: 0 0 40px rgba(59, 130, 246, 0.6), inset 0 0 20px rgba(147, 197, 253, 0.4);
          z-index: 10;
          animation: core-pulse 3s ease-in-out infinite;
        }

        /* Outer Spinning Ring */
        .orb-ring-outer {
          position: absolute;
          width: 180px;
          height: 180px;
          border-radius: 50%;
          border: 2px solid transparent;
          border-top-color: rgba(96, 165, 250, 0.6);
          border-right-color: rgba(59, 130, 246, 0.3);
          box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
          animation: spin 8s linear infinite;
        }

        /* Inner Spinning Ring */
        .orb-ring-inner {
          position: absolute;
          width: 140px;
          height: 140px;
          border-radius: 50%;
          border: 2px solid transparent;
          border-bottom-color: rgba(167, 139, 250, 0.8);
          border-left-color: rgba(139, 92, 246, 0.3);
          animation: spin-reverse 5s linear infinite;
        }

        .agent-status-text {
          font-size: 1.1rem;
          color: #e2e8f0;
          font-weight: 300;
          letter-spacing: 0.5px;
          animation: fadePulse 3s infinite;
        }

        /* --- CUSTOM CONTROLS --- */
        .custom-voice-controls {
          display: flex;
          gap: 1.5rem;
          margin-bottom: 1rem;
        }

        .control-btn {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          border: none;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
          background: rgba(30, 41, 59, 0.6);
          border: 1px solid rgba(148, 163, 184, 0.2);
          color: white;
          backdrop-filter: blur(10px);
        }

        .control-btn:hover {
          transform: translateY(-4px);
          background: rgba(51, 65, 85, 0.8);
        }

        .control-btn.muted {
          background: rgba(255, 255, 255, 0.1);
          color: #94a3b8;
        }

        .control-btn.hangup {
          background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
          box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        }
        
        .control-btn.hangup:hover {
           box-shadow: 0 8px 20px rgba(239, 68, 68, 0.4);
        }

        /* --- KEYFRAMES --- */
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @keyframes spin-reverse {
          0% { transform: rotate(360deg); }
          100% { transform: rotate(0deg); }
        }

        @keyframes core-pulse {
          0%, 100% { transform: scale(1); opacity: 0.9; box-shadow: 0 0 40px rgba(59, 130, 246, 0.6); }
          50% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 60px rgba(96, 165, 250, 0.8); }
        }
        
        @keyframes pulse-red {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes fadePulse {
           0%, 100% { opacity: 0.6; }
           50% { opacity: 1; }
        }
        
        /* Rest of existing styles... */

        @keyframes rotateBg {
          0% { background: linear-gradient(0deg, #0d1b2a 0%, #456586ff 100%); }
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
          background: transparent;
          box-shadow: none;
          border-radius: 0;
        }

        .avatar-logo {
          width: 60px;
          height: 60px;
          object-fit: contain;
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

        .message-bubble strong {
          font-weight: 600;
          color: #60a5fa;
        }

        .message-bubble.user strong {
          color: #fef3c7;
        }

        .message-bubble p {
          margin: 0;
          margin-bottom: 0.5rem;
        }

        .message-bubble p:last-child {
          margin-bottom: 0;
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

        .typing-indicator {
          display: flex;
          gap: 4px;
          align-items: center;
          padding: 0.5rem 0;
        }

        .typing-indicator span {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: currentColor;
          opacity: 0.5;
          animation: typingBounce 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes typingBounce {
          0%, 60%, 100% {
            opacity: 0.5;
            transform: translateY(0);
          }
          30% {
            opacity: 1;
            transform: translateY(-10px);
          }
        }

        .badge.sources {
          background: rgba(168, 85, 247, 0.15);
          color: #d8b4fe;
          border: 1px solid rgba(216, 180, 254, 0.3);
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

        /* Voice Chat Button */
        .voice-btn {
          position: absolute;
          right: 2rem;
          padding: 0.75rem 1.25rem;
          background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
          border: none;
          border-radius: 10px;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
          box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }

        .voice-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
        }

        .voice-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        /* Voice Chat Modal Overlay */
        .voice-chat-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          backdrop-filter: blur(8px);
        }

        .voice-chat-modal {
          background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%);
          border: 1px solid rgba(148, 163, 184, 0.2);
          border-radius: 20px;
          width: 90%;
          max-width: 450px;
          height: 500px;
          display: flex;
          flex-direction: column;
          box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px);
          overflow: hidden;
        }

        /* --- REACTIVE STATES --- */

        /* 1. SPEAKING (Agent is talking) - Lights up bright Gold/Orange */
        .orb-container.speaking .orb-core {
          background: radial-gradient(circle at 30% 30%, rgba(251, 191, 36, 0.4), rgba(245, 158, 11, 0.9));
          box-shadow: 0 0 60px rgba(245, 158, 11, 0.6);
          animation: pulse-speaking 0.5s ease-in-out infinite alternate;
        }
        .orb-container.speaking .orb-ring-outer {
          border-top-color: #fbbf24;
          animation-duration: 2s; /* Spin fast */
        }
        .orb-container.speaking .orb-ring-inner {
          border-bottom-color: #f59e0b;
          animation-duration: 2s;
        }

        /* 2. LISTENING (User is talking) - Blue Pulse */
        .orb-container.listening .orb-core {
          background: radial-gradient(circle at 30% 30%, rgba(59, 130, 246, 0.3), rgba(37, 99, 235, 0.8));
          transform: scale(0.95);
          box-shadow: 0 0 30px rgba(37, 99, 235, 0.5);
        }
        .orb-container.listening .orb-ring-outer {
          border-top-color: #3b82f6;
          animation-duration: 4s;
        }

        /* 3. THINKING (Silence) - Gentle Purple Breathe */
        .orb-container.thinking .orb-core {
          background: radial-gradient(circle at 30% 30%, rgba(139, 92, 246, 0.2), rgba(124, 58, 237, 0.6));
          animation: core-pulse 3s ease-in-out infinite;
        }

        /* Text status color changes */
        .agent-status-text.speaking { color: #fbbf24; text-shadow: 0 0 10px rgba(251, 191, 36, 0.3); }

        @keyframes pulse-speaking {
          from { transform: scale(1); opacity: 0.8; }
          to { transform: scale(1.15); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

// Wrapper to handle props for controls inside LiveKit Room
const CustomControls = ({ handleEndVoiceChat }: { handleEndVoiceChat: () => void }) => {
   return <CustomVoiceControls onDisconnect={handleEndVoiceChat} />
}