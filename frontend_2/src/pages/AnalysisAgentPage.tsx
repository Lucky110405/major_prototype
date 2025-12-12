import { useState, useEffect, useRef } from 'react';
import { BotMessageSquare } from 'lucide-react';
import ChatInterface from '../components/ChatInterface';
import { conversationsApi, messagesApi } from '../lib/api';
import type { Conversation, Message } from '../lib/types';

export default function AnalysisAgentPage() {
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [initialQuery, setInitialQuery] = useState('');
  const [hasStarted, setHasStarted] = useState(false);
  const [workflowResult, setWorkflowResult] = useState<any>(null);
  const streamControllerRef = useRef<{ stop: () => void } | null>(null);

  useEffect(() => {
    if (currentConversation) {
      loadMessages();
    }
  }, [currentConversation]);

  const loadMessages = async () => {
    if (!currentConversation) return;
    const conversationId = (currentConversation as any)?.id ?? (currentConversation as any)?.conversation_id ?? (typeof currentConversation === 'string' ? currentConversation : undefined);
    if (!conversationId) return;
    try {
      const msgs = await messagesApi.getByConversation(conversationId);
      setMessages(msgs);
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  };

  const handleStartAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!initialQuery.trim()) return;

    setIsLoading(true);
    setHasStarted(true);

    try {
      const conversation = await conversationsApi.create(
        initialQuery.slice(0, 50) + (initialQuery.length > 50 ? '...' : '')
      );
      // Ensure conversation has an id (support string return types and wrappers)
      const convId = (conversation as any)?.id ?? (conversation as any)?.conversation_id ?? (typeof conversation === 'string' ? conversation : undefined);
      const normalizedConversation = convId ? { ...(conversation as any), id: convId } as Conversation : (conversation as Conversation);
      if (!normalizedConversation || !normalizedConversation.id) {
        console.error('Conversation creation returned no id. Server response:', conversation);
        throw new Error('Conversation id missing from server response; see console for the raw server response.');
      }
      setCurrentConversation(normalizedConversation);

      const userMessage = await messagesApi.create(
        normalizedConversation.id,
        'user',
        initialQuery.trim()
      );

      // Start streaming response and update assistant placeholder
      const assistantId = `stream-${Date.now()}`;
      const assistantPlaceholder: Message = {
        id: assistantId,
        conversation_id: normalizedConversation.id,
        role: 'assistant',
        content: '',
        created_at: new Date().toISOString(),
      };
      setMessages([userMessage, assistantPlaceholder]);

      // Stop any existing stream
      if (streamControllerRef.current) {
        try { streamControllerRef.current.stop(); } catch (e) {}
      }
      const controller = messagesApi.generateAgentResponseStream(initialQuery.trim(), normalizedConversation.id, async (event: any) => {
        if (event.type === 'partial') {
          // append partial chunk to placeholder message
          setMessages((prev) => prev.map(m => m.id === assistantId ? { ...m, content: (m.content || '') + event.partial } : m));
        } else if (event.type === 'status') {
          // optional: show status indicator
        } else if (event.type === 'final') {
          // Use persisted assistant_message if provided; otherwise update placeholder content
          const assistant_msg = event.assistant_message;
          const result = event.result || event.workflow_result || event;
          if (assistant_msg) {
            setMessages((prev) => prev.map(m => m.id === assistantId ? assistant_msg : m));
          } else {
            setMessages((prev) => prev.map(m => m.id === assistantId ? { ...m, content: result?.final_output || m.content } : m));
          }
          streamControllerRef.current = controller;
          setWorkflowResult(result || null);
          if (event.conversation_id && normalizedConversation.id !== event.conversation_id) {
            setCurrentConversation({ ...normalizedConversation, id: event.conversation_id } as Conversation);
          }
        } else if (event.type === 'error') {
          console.error('Stream error', event.error);
          setMessages((prev) => prev.map(m => m.id === assistantId ? { ...m, content: `Error: ${event.error || 'An error occurred during analysis.'}` } : m));
        }
      });
    } catch (error) {
      console.error('Failed to start analysis:', error);
      const errMsg = (error as any)?.message || 'Failed to start analysis. Please try again.';
      alert(errMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!currentConversation) return;

    setIsLoading(true);

    try {
      const conversationId = (currentConversation as any)?.id ?? (currentConversation as any)?.conversation_id ?? (typeof currentConversation === 'string' ? currentConversation : undefined);
      if (!conversationId) throw new Error('No conversation id');

      const userMessage = await messagesApi.create(
        conversationId,
        'user',
        content
      );

      setMessages((prev) => [...prev, userMessage]);

      // Start streaming response for the sent message
      const assistantId = `stream-${Date.now()}`;
      const assistantPlaceholder: Message = {
        id: assistantId,
        conversation_id: conversationId,
        role: 'assistant',
        content: '',
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantPlaceholder]);

      // Stop any existing stream
      if (streamControllerRef.current) {
        try { streamControllerRef.current.stop(); } catch (e) {}
      }
      const controller = messagesApi.generateAgentResponseStream(content, conversationId, async (event: any) => {
        if (event.type === 'partial') {
          setMessages((prev) => prev.map(m => m.id === assistantId ? { ...m, content: (m.content || '') + event.partial } : m));
        } else if (event.type === 'final') {
          const assistant_msg = event.assistant_message;
          const result = event.result || event.workflow_result || event;
          if (assistant_msg) {
            setMessages((prev) => prev.map(m => m.id === assistantId ? assistant_msg : m));
          } else {
            setMessages((prev) => prev.map(m => m.id === assistantId ? { ...m, content: result?.final_output || m.content } : m));
          }
          setWorkflowResult(result || null);
          if (event.conversation_id && (currentConversation as any)?.id !== event.conversation_id) {
            setCurrentConversation({ ...(currentConversation as any), id: event.conversation_id } as Conversation);
          }
        } else if (event.type === 'error') {
          console.error('Stream error', event.error);
          setMessages((prev) => prev.map(m => m.id === assistantId ? { ...m, content: `Error: ${event.error || 'An error occurred during analysis.'}` } : m));
        }
        });
        streamControllerRef.current = controller;
    } catch (error) {
      console.error('Failed to send message:', error);
      alert('Failed to send message. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (streamControllerRef.current) {
        try { streamControllerRef.current.stop(); } catch (e) {}
        streamControllerRef.current = null;
      }
    };
  }, []);

  const handleNewConversation = () => {
    setCurrentConversation(null);
    setMessages([]);
    setInitialQuery('');
    setHasStarted(false);
  };

  if (!hasStarted) {
    return (
      <div>
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Analysis Agent</h2>
          <p className="text-gray-600">
            Interact with the intelligent agent for deep analysis and multi-turn conversations
          </p>
        </div>

        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-lg border border-gray-200 p-8">
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
                <BotMessageSquare className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Start Your Analysis
              </h3>
              <p className="text-gray-600">
                Ask the agent to analyze or perform a task based on your knowledge base
              </p>
            </div>

            <form onSubmit={handleStartAnalysis}>
              <label htmlFor="initial-query" className="block text-sm font-medium text-gray-700 mb-3">
                What would you like to analyze?
              </label>
              <textarea
                id="initial-query"
                value={initialQuery}
                onChange={(e) => setInitialQuery(e.target.value)}
                placeholder="Ask the agent to analyze or perform a task..."
                rows={4}
                disabled={isLoading}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500 resize-none"
              />
              <button
                type="submit"
                disabled={!initialQuery.trim() || isLoading}
                className="w-full mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {isLoading ? 'Starting Analysis...' : 'Analyze'}
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Analysis Agent</h2>
          <p className="text-gray-600">
            {currentConversation?.title} {currentConversation?.id ? ` — ID: ${currentConversation.id}` : ''}
          </p>
        </div>
        <button
          onClick={handleNewConversation}
          className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium"
        >
          New Conversation
        </button>
      </div>

      <div className="flex-1 bg-white rounded-lg border border-gray-200 overflow-hidden">
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          workflowResult={workflowResult}
        />
      </div>
    </div>
  );
}
