import type { Document, Query, Conversation, Message } from './types';

const API_BASE_URL = 'http://localhost:8000';

export const documentsApi = {
  async getAll(): Promise<Document[]> {
    const response = await fetch(`${API_BASE_URL}/documents`);
    if (!response.ok) {
      // Try to parse error message
      const errText = await response.text();
      throw new Error(`Failed to fetch documents: ${response.status} ${errText}`);
    }
    const result = await response.json();
    // Prefer 'documents' array if present, otherwise if result itself is an array return it.
    if (Array.isArray(result.documents)) {
      return result.documents;
    }
    if (Array.isArray(result)) {
      return result;
    }
    // If the backend returned an object but not a list, return empty array for safety
    return [];
  },

  async upload(file: File): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/ingest/auto`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error('Failed to upload document');
    const result = await response.json();
    return result.document || result;
  },

  async delete(id: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/documents/${id}`, {
      method: 'DELETE',
    });

    if (!response.ok) throw new Error('Failed to delete document');
  },
};

export const queriesApi = {
  async create(queryText: string): Promise<{ query: Query; sources: Document[] }> {
    const response = await fetch(`${API_BASE_URL}/query?q=${encodeURIComponent(queryText)}`);
    if (!response.ok) throw new Error('Failed to create query');
    const result = await response.json();
    return { query: result.query, sources: result.sources || [] };
  },
};

export const conversationsApi = {
  async create(title?: string): Promise<Conversation> {
    const response = await fetch(`${API_BASE_URL}/agents/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });

    if (!response.ok) throw new Error('Failed to create conversation');
    const result = await response.json();
    let conv = result.conversation || result;
    // If we got a wrapper response with a data object, unwrap
    if (conv && (conv as any).data) {
      conv = (conv as any).data;
    }
    // If conv is an array, pick the first entry
    if (Array.isArray(conv) && conv.length > 0) {
      conv = conv[0];
    }
    // If the backend returned a minimal wrapper like {value: {...}}
    if (conv && typeof conv === 'object') {
      // Try to find an id in nested objects
      const tryGetId = (obj: any): any => {
        if (!obj) return null;
        if (typeof obj === 'string') return obj;
        if (obj.id) return obj.id;
        if (obj.conversation_id) return obj.conversation_id;
        if (obj.data) return tryGetId(obj.data);
        if (Array.isArray(obj)) return tryGetId(obj[0]);
        // search in properties
        for (const k of Object.keys(obj)) {
          const v = (obj as any)[k];
          if (v && typeof v === 'object') {
            const nested = tryGetId(v);
            if (nested) return nested;
          }
        }
        return null;
      };
      const id = tryGetId(conv);
      if (id && !((conv as any).id)) {
        (conv as any).id = id;
      }
    }
    // Normalization: support several shapes that might come from backend or Supabase
    if (typeof conv === 'string') {
      conv = { id: conv, title: title || '' } as any;
    }
    if (Array.isArray(conv) && conv.length > 0) {
      conv = conv[0];
    }
    if (conv && !conv.id && conv.data) {
      // Supabase might wrap data
      conv = conv.data;
    }
    // More defensive return: if no id present, log the raw result for debugging.
    if (!conv || !(conv as Conversation).id) {
      console.warn('conversationsApi.create: returned conversation without id', { rawResult: result, normalized: conv });
    }
    return conv as Conversation;
  },

  async getAll(): Promise<Conversation[]> {
    const response = await fetch(`${API_BASE_URL}/agents/conversations`);
    if (!response.ok) throw new Error('Failed to fetch conversations');
    const result = await response.json();
    return result.conversations || result;
  },
};

export const messagesApi = {
  async getByConversation(conversationId?: string | null): Promise<Message[]> {
    if (!conversationId) return [];
    const response = await fetch(`${API_BASE_URL}/agents/messages?conversation_id=${encodeURIComponent(conversationId)}`);
    if (!response.ok) throw new Error('Failed to fetch messages');
    const result = await response.json();
    return result.messages || result;
  },

  async create(conversationId: string | undefined, role: 'user' | 'assistant', content: string): Promise<Message> {
    if (!conversationId) throw new Error('conversationId is required');
    const response = await fetch(`${API_BASE_URL}/agents/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: conversationId, role, content }),
    });

    if (!response.ok) throw new Error('Failed to create message');
    const result = await response.json();
    return result.message || result;
  },

  async generateAgentResponse(userMessage: string, conversationId?: string | null): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/agents/messages/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: conversationId, content: userMessage }),
    });
    if (!response.ok) throw new Error('Failed to generate response');
    const result = await response.json();
    return result; // { assistant_message: { id, conversation_id, role, content }, workflow_result: {...} }
  },

  generateAgentResponseStream(userMessage: string, conversationId?: string | null, onEvent?: (event: any) => void): { stop: () => void } {
    const controller = new AbortController();
    const signal = controller.signal;
    (async () => {
      try {
          const res = await fetch(`${API_BASE_URL}/agents/messages/generate/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ conversation_id: conversationId, content: userMessage }),
          signal,
        });
          if (!res.ok) {
            const txt = await res.text();
            if (onEvent) onEvent({ type: 'error', error: `${res.status} ${txt}` });
            return;
          }
        const reader = res.body?.getReader();
        if (!reader) return;
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let parts = buffer.split('\n\n');
          buffer = parts.pop() || '';
          for (const part of parts) {
            if (!part.trim()) continue;
            let raw = part;
            if (raw.startsWith('data: ')) raw = raw.substring(6);
            try {
              const event = JSON.parse(raw);
              if (onEvent) onEvent(event);
            } catch (e) {
              // not JSON
            }
          }
        }
      } catch (err) {
        if (onEvent) onEvent({ type: 'error', error: (err as any)?.message || String(err) });
      }
    })();

    return { stop: () => controller.abort() };
  },
};
