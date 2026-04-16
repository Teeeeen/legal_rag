import { useState } from 'react'
import { MessageSquare, Database, Activity } from 'lucide-react'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import PerformancePage from './pages/PerformancePage'

const NAV_ITEMS = [
  { key: 'chat', label: '智能问答', icon: MessageSquare },
  { key: 'knowledge', label: '知识库管理', icon: Database },
  { key: 'performance', label: '性能监控', icon: Activity },
]

export default function App() {
  const [page, setPage] = useState('chat')

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="icon">&#9878;</div>
            法律RAG系统
          </div>
        </div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <button
              key={item.key}
              className={`nav-item ${page === item.key ? 'active' : ''}`}
              onClick={() => setPage(item.key)}
            >
              <item.icon size={18} />
              {item.label}
            </button>
          ))}
        </nav>
        <div style={{ padding: '16px 20px', borderTop: '1px solid rgba(255,255,255,0.1)', fontSize: 11, color: 'var(--text-muted)' }}>
          Qwen3:8B + BGE-M3<br />
          LangChain + ChromaDB
        </div>
      </aside>
      <main className="main-content">
        {/* 用 display 控制显隐，避免切换 tab 时丢失 ChatPage 状态 */}
        <div style={{ display: page === 'chat' ? 'block' : 'none', height: '100%' }}>
          <ChatPage />
        </div>
        <div style={{ display: page === 'knowledge' ? 'block' : 'none' }}>
          <KnowledgePage />
        </div>
        <div style={{ display: page === 'performance' ? 'block' : 'none' }}>
          <PerformancePage />
        </div>
      </main>
    </div>
  )
}
