import { useState } from 'react';
import { Search } from 'lucide-react';
import SourceList from '../components/SourceList';
import { queriesApi } from '../lib/api';
import type { Document } from '../lib/types';

export default function QueryPage() {
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<Document[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query.trim()) return;

    setIsQuerying(true);
    setAnswer(null);
    setSources([]);

    try {
      const result = await queriesApi.create(query.trim());
      if (!result || !result.query) {
        throw new Error('Invalid query response');
      }
      setAnswer(result.query.answer ?? '');
      setSources(Array.isArray(result.sources) ? result.sources : []);
    } catch (error) {
      console.error('Failed to execute query:', error);
      alert('Failed to execute query. Please try again.');
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Query Interface</h2>
        <p className="text-gray-600">
          Ask questions based on your ingested documents
        </p>
      </div>

      <div className="space-y-6">
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <form onSubmit={handleSubmit}>
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-3">
              Your Question
            </label>
            <div className="flex gap-3">
              <input
                id="query"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask anything based on your ingested sources..."
                disabled={isQuerying}
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-500"
              />
              <button
                type="submit"
                disabled={!query.trim() || isQuerying}
                className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2 font-medium"
              >
                <Search className="w-5 h-5" />
                <span>{isQuerying ? 'Querying...' : 'Query'}</span>
              </button>
            </div>
          </form>
        </div>

        {isQuerying && (
          <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
            <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Processing your query...</p>
          </div>
        )}

        {answer && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg border border-gray-200">
              <div className="p-6 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Answer</h3>
              </div>
              <div className="p-6">
                <div className="prose prose-sm max-w-none">
                  <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
                    {answer}
                  </p>
                </div>
              </div>
            </div>

            <div>
              <SourceList
                documents={sources}
                title="Retrieved Sources"
              />
            </div>
          </div>
        )}

        {!answer && !isQuerying && (
          <div className="bg-gray-50 rounded-lg border border-gray-200 p-12 text-center">
            <Search className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Ready to Answer Your Questions
            </h3>
            <p className="text-gray-600">
              Enter your query above to retrieve relevant information from your documents
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
