import { useState, useEffect, useCallback } from 'react'
import { Activity, Play, Download, FileText, CheckSquare } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { getSystemInfo, runBenchmark, createReport, listReports, getReportDownloadUrl } from '../services/api'

export default function PerformancePage() {
  const [sysInfo, setSysInfo] = useState(null)
  const [benchResult, setBenchResult] = useState(null)
  const [running, setRunning] = useState(false)
  const [evaluateQuality, setEvaluateQuality] = useState(false)
  const [reports, setReports] = useState([])
  const [exporting, setExporting] = useState(false)

  const refreshSysInfo = useCallback(async () => {
    try {
      const info = await getSystemInfo()
      setSysInfo(info)
    } catch {
      /* ignore */
    }
  }, [])

  const refreshReports = useCallback(async () => {
    try {
      const list = await listReports()
      setReports(list || [])
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    refreshSysInfo()
    refreshReports()
    const timer = setInterval(refreshSysInfo, 5000)
    return () => clearInterval(timer)
  }, [refreshSysInfo, refreshReports])

  const handleBenchmark = async () => {
    setRunning(true)
    try {
      const result = await runBenchmark(null, true, evaluateQuality)
      setBenchResult(result)
    } catch (err) {
      alert(`测试失败: ${err.response?.data?.detail || err.message}`)
    } finally {
      setRunning(false)
    }
  }

  const handleExportReport = async () => {
    setExporting(true)
    try {
      const report = await createReport(null, true, true)
      setBenchResult(report.benchmark || report)
      await refreshReports()
      alert('报告已生成并保存')
    } catch (err) {
      alert(`导出失败: ${err.response?.data?.detail || err.message}`)
    } finally {
      setExporting(false)
    }
  }

  const chartData = benchResult?.details
    ?.filter(d => !d.error)
    ?.map((d, i) => ({
      name: `Q${i + 1}`,
      query: d.query?.slice(0, 20) + '...',
      retrieval: d.retrieval_ms || 0,
      generation: d.generation_ms || 0,
      total: d.latency_ms || 0,
    })) || []

  const quality = benchResult?.quality

  return (
    <div>
      <h2 style={{ marginBottom: 20 }}>性能监控</h2>

      {sysInfo && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{sysInfo.cpu_percent}%</div>
            <div className="stat-label">CPU 使用率</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{sysInfo.memory_percent}%</div>
            <div className="stat-label">内存使用率</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{sysInfo.memory_used_gb}</div>
            <div className="stat-label">已用内存 (GB)</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{sysInfo.memory_total_gb}</div>
            <div className="stat-label">总内存 (GB)</div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Activity size={18} /> 基准测试
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={evaluateQuality}
                onChange={e => setEvaluateQuality(e.target.checked)}
                disabled={running}
              />
              <CheckSquare size={14} /> 质量评估
            </label>
            <button className="btn btn-primary" onClick={handleBenchmark} disabled={running || exporting}>
              {running ? <><span className="spinner" /> 测试中...</> : <><Play size={14} /> 运行测试</>}
            </button>
            <button
              className="btn"
              onClick={handleExportReport}
              disabled={exporting || running}
              style={{ background: '#6366f1', color: '#fff' }}
            >
              {exporting ? <><span className="spinner" /> 生成中...</> : <><Download size={14} /> 导出报告</>}
            </button>
          </div>
        </div>

        {!benchResult && !running && (
          <div className="empty-state" style={{ padding: 30 }}>
            <p>点击"运行测试"执行基准性能测试，或"导出报告"生成含质量评估的完整报告</p>
          </div>
        )}

        {benchResult && (
          <>
            <div className="stats-grid" style={{ marginTop: 12 }}>
              <div className="stat-card">
                <div className="stat-value">{benchResult.total_queries}</div>
                <div className="stat-label">查询数</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{benchResult.avg_latency_ms}</div>
                <div className="stat-label">平均延迟 (ms)</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{benchResult.avg_retrieval_ms}</div>
                <div className="stat-label">平均检索 (ms)</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{benchResult.queries_per_second}</div>
                <div className="stat-label">QPS</div>
              </div>
            </div>

            {/* 质量指标卡片 */}
            {quality && (
              <div style={{ marginTop: 16 }}>
                <h4 style={{ fontSize: 14, marginBottom: 8 }}>质量评估指标</h4>
                <div className="stats-grid">
                  <div className="stat-card" style={{ borderLeft: '3px solid #8b5cf6' }}>
                    <div className="stat-value">{quality.avg_rouge_l?.toFixed(4) ?? '-'}</div>
                    <div className="stat-label">ROUGE-L (F1)</div>
                  </div>
                  <div className="stat-card" style={{ borderLeft: '3px solid #3b82f6' }}>
                    <div className="stat-value">{quality.avg_retrieval_relevance?.toFixed(2) ?? '-'}</div>
                    <div className="stat-label">检索相关性 (0-10)</div>
                  </div>
                  <div className="stat-card" style={{ borderLeft: '3px solid #22c55e' }}>
                    <div className="stat-value">{quality.avg_faithfulness?.toFixed(2) ?? '-'}</div>
                    <div className="stat-label">忠实度 (0-10)</div>
                  </div>
                  <div className="stat-card" style={{ borderLeft: '3px solid #f59e0b' }}>
                    <div className="stat-value">{quality.avg_rouge_1?.toFixed(4) ?? '-'}</div>
                    <div className="stat-label">ROUGE-1 (F1)</div>
                  </div>
                </div>
              </div>
            )}

            {chartData.length > 0 && (
              <div style={{ marginTop: 20 }}>
                <h4 style={{ fontSize: 14, marginBottom: 12 }}>各查询延迟分解 (ms)</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData}>
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null
                        const d = payload[0].payload
                        return (
                          <div style={{ background: '#fff', padding: 10, border: '1px solid #e2e8f0', borderRadius: 6, fontSize: 12 }}>
                            <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.query}</div>
                            <div>检索: {d.retrieval}ms</div>
                            <div>生成: {d.generation}ms</div>
                            <div>总计: {d.total}ms</div>
                          </div>
                        )
                      }}
                    />
                    <Bar dataKey="retrieval" stackId="a" fill="#3b82f6" name="检索" />
                    <Bar dataKey="generation" stackId="a" fill="#22c55e" name="生成" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div style={{ marginTop: 16 }}>
              <h4 style={{ fontSize: 14, marginBottom: 8 }}>详细结果</h4>
              {benchResult.details.map((d, i) => (
                <div key={i} className="file-item" style={{ marginBottom: 6 }}>
                  <div className="file-info">
                    <span className="file-name" style={{ fontSize: 13 }}>{d.query}</span>
                    {d.error ? (
                      <span className="file-meta" style={{ color: 'var(--danger)' }}>错误: {d.error}</span>
                    ) : (
                      <span className="file-meta">
                        延迟 {d.latency_ms}ms | 检索 {d.retrieval_ms}ms | 生成 {d.generation_ms}ms | 来源 {d.sources_count} 条
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* 报告历史 */}
      <div className="card" style={{ marginTop: 20 }}>
        <div className="card-header">
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FileText size={18} /> 历史报告
          </span>
        </div>

        {reports.length === 0 ? (
          <div className="empty-state" style={{ padding: 20 }}>
            <p>暂无报告，点击"导出报告"生成第一份报告</p>
          </div>
        ) : (
          <div>
            {reports.map((r) => (
              <div key={r.report_id} className="file-item" style={{ marginBottom: 6 }}>
                <div className="file-info" style={{ flex: 1 }}>
                  <span className="file-name" style={{ fontSize: 13 }}>
                    {r.created_at?.replace('T', ' ').slice(0, 19)}
                  </span>
                  <span className="file-meta">
                    查询 {r.total_queries} 条 | 延迟 {r.avg_latency_ms}ms | QPS {r.queries_per_second}
                    {r.has_quality && ' | 含质量评估'}
                  </span>
                </div>
                <a
                  href={getReportDownloadUrl(r.report_id)}
                  download
                  className="btn"
                  style={{ fontSize: 12, padding: '4px 10px' }}
                >
                  <Download size={12} /> 下载
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
