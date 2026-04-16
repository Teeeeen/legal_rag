import { useState, useEffect, useCallback } from 'react'
import { Upload, Trash2, RefreshCw, FileText } from 'lucide-react'
import { listDocuments, uploadDocument, deleteDocument, rebuildIndex, getKBStats } from '../services/api'

export default function KnowledgePage() {
  const [files, setFiles] = useState([])
  const [stats, setStats] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [docType, setDocType] = useState('law')

  const refresh = useCallback(async () => {
    try {
      const [fileList, kbStats] = await Promise.all([listDocuments(), getKBStats()])
      setFiles(fileList)
      setStats(kbStats)
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      await uploadDocument(file, docType)
      await refresh()
    } catch (err) {
      alert(`上传失败: ${err.response?.data?.detail || err.message}`)
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  const handleDelete = async (filename) => {
    if (!confirm(`确定删除 ${filename}？`)) return
    try {
      await deleteDocument(filename)
      await refresh()
    } catch (err) {
      alert(`删除失败: ${err.response?.data?.detail || err.message}`)
    }
  }

  const handleRebuild = async () => {
    if (!confirm('重建索引会清除现有向量数据并重新导入，确定继续？')) return
    setRebuilding(true)
    try {
      await rebuildIndex()
      await refresh()
    } catch (err) {
      alert(`重建失败: ${err.response?.data?.detail || err.message}`)
    } finally {
      setRebuilding(false)
    }
  }

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  return (
    <div>
      <h2 style={{ marginBottom: 20 }}>知识库管理</h2>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{stats.total_files}</div>
            <div className="stat-label">文件总数</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.total_chunks}</div>
            <div className="stat-label">总分块数</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.laws_chunks}</div>
            <div className="stat-label">法律条文分块</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.cases_chunks}</div>
            <div className="stat-label">指导案例分块</div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <Upload size={18} /> 上传文档
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <select className="select-small" value={docType} onChange={e => setDocType(e.target.value)}>
            <option value="law">法律条文</option>
            <option value="case">指导案例</option>
          </select>
          <label className="btn btn-primary" style={{ cursor: uploading ? 'not-allowed' : 'pointer' }}>
            {uploading ? <><span className="spinner" /> 上传中...</> : <><Upload size={14} /> 选择文件</>}
            <input type="file" accept=".txt,.md,.json" onChange={handleUpload} disabled={uploading} style={{ display: 'none' }} />
          </label>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>支持 .txt, .md, .json 格式</span>
        </div>
      </div>

      <div className="card">
        <div className="card-header" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FileText size={18} /> 文件列表
          </span>
          <button className="btn btn-outline" onClick={handleRebuild} disabled={rebuilding}>
            {rebuilding ? <><span className="spinner" /> 重建中...</> : <><RefreshCw size={14} /> 重建索引</>}
          </button>
        </div>

        {files.length === 0 ? (
          <div className="empty-state" style={{ padding: 30 }}>
            <p>知识库为空，请上传法律文档</p>
          </div>
        ) : (
          <div className="file-list">
            {files.map((f, i) => (
              <div key={i} className="file-item">
                <div className="file-info">
                  <span className="file-name">{f.filename}</span>
                  <span className="file-meta">
                    {f.doc_type === 'law' ? '法律条文' : '指导案例'} | {formatSize(f.size_bytes)}
                  </span>
                </div>
                <button className="btn btn-danger" style={{ padding: '4px 10px' }} onClick={() => handleDelete(f.filename)}>
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
