import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

// ===================== 问答 =====================

export async function sendChat({
  question,
  useRerank = true,
  useQueryRewrite = false,
  topK = 5,
  collection = 'all',
  evaluateQuality = false,
  monitorSystem = false,
  queryTransform = 'none',
  rerankStrategy = 'simple',
  generationStrategy = 'standard',
  useKg = false,
}) {
  const res = await api.post('/chat', {
    question,
    use_rerank: useRerank,
    use_query_rewrite: useQueryRewrite,
    top_k: topK,
    collection,
    evaluate_quality: evaluateQuality,
    monitor_system: monitorSystem,
    query_transform: queryTransform,
    rerank_strategy: rerankStrategy,
    generation_strategy: generationStrategy,
    use_kg: useKg,
  })
  return res.data.data
}

// ===================== 知识库 =====================

export async function uploadDocument(file, docType) {
  const form = new FormData()
  form.append('file', file)
  if (docType) form.append('doc_type', docType)
  const res = await api.post('/knowledge/upload', form)
  return res.data.data
}

export async function listDocuments() {
  const res = await api.get('/knowledge/list')
  return res.data.data
}

export async function deleteDocument(filename) {
  const res = await api.delete(`/knowledge/${encodeURIComponent(filename)}`)
  return res.data
}

export async function rebuildIndex() {
  const res = await api.post('/knowledge/rebuild')
  return res.data.data
}

export async function getKBStats() {
  const res = await api.get('/knowledge/stats')
  return res.data.data
}

// ===================== 性能 =====================

export async function getSystemInfo() {
  const res = await api.get('/performance/system')
  return res.data.data
}

export async function runBenchmark(queries, useRerank = true, evaluateQuality = false) {
  const res = await api.post('/performance/bench', null, {
    params: { use_rerank: useRerank, evaluate_quality: evaluateQuality },
  })
  return res.data.data
}

// ===================== 报告 =====================

export async function createReport(queries, useRerank = true, evaluateQuality = true) {
  const res = await api.post('/performance/report', null, {
    params: { use_rerank: useRerank, evaluate_quality: evaluateQuality },
  })
  return res.data.data
}

export async function listReports() {
  const res = await api.get('/performance/reports')
  return res.data.data
}

export function getReportDownloadUrl(reportId) {
  return `/api/performance/reports/${encodeURIComponent(reportId)}`
}

// ===================== 问答记录 =====================

export async function saveChatRecord(record) {
  const res = await api.post('/chat/save-record', record)
  return res.data.data
}

export async function listChatRecords() {
  const res = await api.get('/chat/records')
  return res.data.data
}

export function getChatRecordDownloadUrl(recordId) {
  return `/api/chat/records/${encodeURIComponent(recordId)}`
}
