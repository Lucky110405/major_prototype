import { ReactNode } from 'react';
import { Upload, Search, BotMessageSquare } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
  currentPage: 'ingest' | 'query' | 'agent';
  onNavigate: (page: 'ingest' | 'query' | 'agent') => void;
}

export default function Layout({ children, currentPage, onNavigate }: LayoutProps) {
  const navItems = [
    { id: 'ingest' as const, label: 'Ingest', icon: Upload },
    { id: 'query' as const, label: 'Query', icon: Search },
    { id: 'agent' as const, label: 'Analysis Agent', icon: BotMessageSquare },
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <h1 className="text-2xl font-bold text-gray-900">CMM Agentic Framework</h1>
          <p className="text-sm text-gray-500 mt-1">Business Intelligence</p>
        </div>

        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;

              return (
                <li key={item.id}>
                  <button
                    onClick={() => onNavigate(item.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 font-medium'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <Icon className={`w-5 h-5 ${isActive ? 'text-blue-700' : 'text-gray-500'}`} />
                    <span>{item.label}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="p-4 border-t border-gray-200">
          <div className="px-4 py-3 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-600">
              Upload documents, query your knowledge base, or interact with the analysis agent.
            </p>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto p-8">
          {children}
        </div>
      </main>
    </div>
  );
}
