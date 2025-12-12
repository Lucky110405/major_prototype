import { FileText, Trash2 } from 'lucide-react';
import type { Document } from '../lib/types';

interface SourceListProps {
  documents: Document[];
  onDelete?: (id: string) => void;
  showDelete?: boolean;
  title?: string;
}

export default function SourceList({
  documents,
  onDelete,
  showDelete = false,
  title = 'Sources'
}: SourceListProps) {
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (documents.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="text-center text-gray-500">
          <FileText className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p>No documents available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <p className="text-sm text-gray-500 mt-1">{documents.length} document(s)</p>
      </div>

      <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
        {documents.map((doc) => (
          <div key={doc.id} className="p-4 hover:bg-gray-50 transition-colors">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3 flex-1 min-w-0">
                <FileText className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-medium text-gray-900 truncate">
                    {doc.file_name}
                  </h4>
                  <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                    <span className="uppercase">{doc.file_type}</span>
                    <span>{formatFileSize(doc.file_size)}</span>
                    <span>{formatDate(doc.uploaded_at)}</span>
                  </div>
                </div>
              </div>

              {showDelete && onDelete && (
                <button
                  onClick={() => onDelete(doc.id)}
                  className="ml-3 p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors flex-shrink-0"
                  title="Delete document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
