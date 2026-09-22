import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { FileText, Upload, Send, Sparkles, BookOpen, X, ChevronRight, LoaderCircle, ArrowUpRight } from 'lucide-react'
import './styles.css'
import './overrides.css'

const API = '/api'

function App() {
  const [documents, setDocuments] = useState([])
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  const refreshDocuments = () => fetch(`${API}/documents`).then(r => r.json()).then(setDocuments).catch(() => {})
  useEffect(() => { refreshDocuments() }, [])

  async function upload(file) {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) return setError('Please choose a PDF file.')
    setUploading(true); setError('')
    const body = new FormData(); body.append('file', file)
    try {
      const r = await fetch(`${API}/upload`, { method: 'POST', body })
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Upload failed')
      await refreshDocuments()
    } catch (e) { setError(e.message) } finally { setUploading(false) }
  }

  async function ask(event) {
    event.preventDefault()
    if (!question.trim() || asking) return
    setAsking(true); setError(''); setAnswer(null)
    try {
      const r = await fetch(`${API}/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) })
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Unable to answer this question')
      setAnswer(data)
    } catch (e) { setError(e.message) } finally { setAsking(false) }
  }

  async function removeDocument(id) {
    await fetch(`${API}/documents/${id}`, { method: 'DELETE' })
    setAnswer(null); refreshDocuments()
  }

  return <main>
    <div className="orb orb-one" /><div className="orb orb-two" /><div className="grid" />
    <nav><a className="brand" href="#"><span className="brand-mark"><Sparkles size={17}/></span>ATLAS</a><span className="nav-label">PDF KNOWLEDGE ASSISTANT</span></nav>
    <section className="hero">
      <h1>Ask more of your<br/><i>documents.</i></h1>
      <p className="subhead">Upload any PDF. Atlas finds the signal, synthesizes the answer, and shows you exactly where it came from.</p>
    </section>
    <section className="workspace">
      <aside className="library">
        <div className="panel-heading"><div><span className="kicker">01 / LIBRARY</span><h2>Knowledge base</h2></div><span className="count">{documents.length}</span></div>
        <input ref={inputRef} className="hidden" type="file" accept="application/pdf" onChange={e => upload(e.target.files?.[0])}/>
        <button className="upload-zone" onClick={() => inputRef.current?.click()} disabled={uploading}>
          {uploading ? <LoaderCircle className="spin"/> : <Upload/>}<strong>{uploading ? 'Indexing document…' : 'Add a PDF'}</strong><small>Up to 25 MB · text-based PDFs</small>
        </button>
        <div className="document-list">{documents.length === 0 ? <p className="empty">Your uploaded documents will live here.</p> : documents.map(doc => <article className="document" key={doc.id}><span className="file-icon"><FileText size={18}/></span><div><strong title={doc.name}>{doc.name}</strong><small>{doc.pages} pages · {doc.chunks} passages</small></div><button onClick={() => removeDocument(doc.id)} aria-label={`Remove ${doc.name}`}><X size={15}/></button></article>)}</div>
      </aside>
      <section className="chat">
        <div className="panel-heading"><div><span className="kicker">02 / QUERY</span><h2>What would you like to know?</h2></div><span className="live"><b /> {documents.length ? 'READY' : 'WAITING FOR SOURCES'}</span></div>
        <form onSubmit={ask} className="ask-box"><textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about your documents…" rows="3"/><button type="submit" disabled={!documents.length || !question.trim() || asking}>{asking ? <LoaderCircle className="spin" size={19}/> : <Send size={19}/>}<span>{asking ? 'Thinking' : 'Ask Atlas'}</span></button></form>
        {error && <div className="error">{error}</div>}
        {!answer && !asking && <div className="empty-state"><div className="empty-icon"><BookOpen size={29}/></div><p>{documents.length ? 'Your cited answer will appear here.' : 'Add a document to start a conversation.'}</p></div>}
        {asking && <div className="thinking"><span/><span/><span/> Searching your knowledge base</div>}
        {answer && <div className="answer"><div className="answer-label"><Sparkles size={15}/> ATLAS RESPONSE <span>{answer.used_llm ? 'SYNTHESIZED' : 'SOURCE EXCERPT'}</span></div><p>{answer.answer}</p><div className="sources"><h3>Sources <small>{answer.citations.length} passages retrieved</small></h3>{answer.citations.map((citation, i) => <article className="source" key={`${citation.document}-${citation.page}-${i}`}><span className="source-no">{String(i+1).padStart(2, '0')}</span><div><strong>{citation.document} <em>· PAGE {citation.page}</em></strong><p>{citation.excerpt}</p></div><ChevronRight size={17}/></article>)}</div></div>}
      </section>
    </section>
    <footer id="how-it-works"><span>ATLAS / RAG SYSTEM</span><p>RETRIEVE <b>→</b> REASON <b>→</b> REFERENCE</p><span>Built for curious minds</span></footer>
  </main>
}
createRoot(document.getElementById('root')).render(<App />)
