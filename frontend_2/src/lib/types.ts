export interface Document {
  id: string;
  file_name: string;
  file_type: string;
  file_size: number;
  content: string | null;
  uploaded_at: string;
  user_id: string | null;
}

export interface Query {
  id: string;
  query_text: string;
  answer: string | null;
  created_at: string;
  user_id: string | null;
}

export interface QuerySource {
  id: string;
  query_id: string;
  document_id: string;
  relevance_score: number;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  user_id: string | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}

export interface Database {
  public: {
    Tables: {
      documents: {
        Row: Document;
        Insert: Omit<Document, 'id' | 'uploaded_at'>;
        Update: Partial<Omit<Document, 'id' | 'uploaded_at'>>;
      };
      queries: {
        Row: Query;
        Insert: Omit<Query, 'id' | 'created_at'>;
        Update: Partial<Omit<Query, 'id' | 'created_at'>>;
      };
      query_sources: {
        Row: QuerySource;
        Insert: Omit<QuerySource, 'id'>;
        Update: Partial<Omit<QuerySource, 'id'>>;
      };
      conversations: {
        Row: Conversation;
        Insert: Omit<Conversation, 'id' | 'created_at'>;
        Update: Partial<Omit<Conversation, 'id' | 'created_at'>>;
      };
      messages: {
        Row: Message;
        Insert: Omit<Message, 'id' | 'created_at'>;
        Update: Partial<Omit<Message, 'id' | 'created_at'>>;
      };
    };
  };
}
