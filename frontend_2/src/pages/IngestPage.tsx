import { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import SourceList from '../components/SourceList';
import { documentsApi } from '../lib/api';
import type { Document } from '../lib/types';

export default function IngestPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await documentsApi.getAll();
      // Ensure we only set an array
      setDocuments(Array.isArray(docs) ? docs : []);
    } catch (error) {
      console.error('Failed to load documents:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    try {
      await documentsApi.upload(file);
      await loadDocuments();
    } catch (error) {
      console.error('Failed to upload document:', error);
      alert('Failed to upload document. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this document?')) {
      return;
    }

    try {
      await documentsApi.delete(id);
      setDocuments(documents.filter((doc) => doc.id !== id));
    } catch (error) {
      console.error('Failed to delete document:', error);
      alert('Failed to delete document. Please try again.');
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Document Ingestion</h2>
        <p className="text-gray-600">
          Upload and manage your documents for the RAG system
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <FileUpload onUpload={handleUpload} isUploading={isUploading} />
        </div>

        <div>
          {isLoading ? (
            <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-gray-600">Loading documents...</p>
            </div>
          ) : (
            <SourceList
              documents={documents}
              onDelete={handleDelete}
              showDelete={true}
              title="Your Documents"
            />
          )}
        </div>
      </div>

      {documents.length > 0 && (
        <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-blue-800">
            <strong>{documents.length}</strong> document(s) ready for querying and analysis
          </p>
        </div>
      )}
    </div>
  );
}
